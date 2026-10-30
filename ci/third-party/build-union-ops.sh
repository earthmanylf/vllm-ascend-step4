#!/bin/bash
# Compile the two union-mode custom op packages and flatten them into prebuilt/.
#
# The op repos are not built inside the image: each needs a full toolchain run,
# and the result is a finished vendor tree. This script does that once, offline,
# and leaves prebuilt/ in the layout ci/build.sh --with-union-ops stages.
#
# Everything it needs comes from this directory (see README.md):
#   ops-transformer/ ops-nn/          the two op repos
#   cann-cmake/ opbase/ ops-tensor/   cmake-time dependencies, pinned per repo
#   catlass/ ops-test-kit/            same
#   assets/                           json/abseil/protobuf/eigen/makeself tarballs;
#                                     taken from the image when absent
#
# Usage:
#   ./build-union-ops.sh [--image REF] [--assets DIR] [--jobs N] [--clean]
#
# --image  image that carries the CANN 9.2.0 toolchain (default: the same base
#          the server runs on, so the packages match the runtime)
# --assets directory with the third-party tarballs; default assets/, and when it
#          is empty they are taken from the image's /opt/step5/ops-thirdparty
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${STEP5_BASE_IMAGE:-step5-ascend950:decode-perf-8ccadfa3bf3b}"
ASSETS="$HERE/assets"
JOBS="$(nproc)"
CLEAN=0
DOCKER="docker"

while (($#)); do
  case "$1" in
    --image)  IMAGE="$2"; shift 2 ;;
    --assets) ASSETS="$2"; shift 2 ;;
    --jobs)   JOBS="$2"; shift 2 ;;
    --clean)  CLEAN=1; shift ;;
    -h|--help) sed -n '2,26p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

for d in ops-transformer ops-nn cann-cmake opbase ops-tensor catlass ops-test-kit; do
  [[ -d "$HERE/$d" ]] || { echo "missing $HERE/$d -- run ./fetch.py first" >&2; exit 1; }
done

# Both op repos check opbase/ops-tensor/cann-cmake out in place, and the pins
# differ between them, so each gets its own copy of the whole dependency set.
stage_deps() {
  local variant="$1" opbase_ref="$2" tensor_ref="$3" cmake_ref="$4"
  local dst="$HERE/.stage/$variant"
  rm -rf "$dst"
  mkdir -p "$dst"
  for d in opbase ops-tensor catlass ops-test-kit; do
    git clone -q --shared "$HERE/$d" "$dst/$d"
  done
  git -C "$dst/opbase" checkout -q "$opbase_ref"
  git -C "$dst/ops-tensor" checkout -q "$tensor_ref"
  git -C "$dst/ops-tensor" submodule update --init --recursive -q
  git -C "$dst/catlass" checkout -q 769cd40a8716b28650b6bebb08db4834eea4462f
  # cann-cmake is a tag per repo, and the op repos look for it by path only.
  mkdir -p "$dst/cann-cmake"
  git -C "$HERE/cann-cmake" archive "$cmake_ref" | tar -x -C "$dst/cann-cmake"
  if [[ -d "$ASSETS" ]] && compgen -G "$ASSETS/*" >/dev/null; then
    cp -a "$ASSETS"/. "$dst/"
  fi
}

echo "==> staging pinned dependencies"
stage_deps transformer \
  0a83b915a6f6be49a03b907713d6f000b3dea1ea 99cc37f124b1b2521f96cb1b03dd143a41ddfa3a master-058
stage_deps nn \
  40cc7bedb324a9358aa247a75f5ff898dd46f18c 6d0ca6e16abfe355586cd7213845cdc1b180e38f master-046

mkdir -p "$HERE/.stage"
cat > "$HERE/.stage/in-container.sh" <<'INNER'
set -euo pipefail
git config --global --add safe.directory "*"
export TMPDIR=/w/.stage/tmp CCACHE_DIR=/w/.stage/ccache
mkdir -p "$TMPDIR" "$CCACHE_DIR"
source /usr/local/Ascend/cann-9.2.0/set_env.sh >/dev/null 2>&1

# The tarballs the vendored cmake find_package calls want (json, abseil, eigen,
# protobuf, makeself). The image already carries them for its own op build, so
# take them from there instead of the network.
for v in transformer nn; do
  d=/w/.stage/$v
  for f in json-3.12.0.tar.gz abseil-cpp abseil-cpp-20230802.1.tar.gz \
           eigen-5.0.0.tar.gz protobuf protobuf-25.1.tar.gz lib_cache pkg; do
    [ -e "$d/$f" ] || cp -a "/opt/step5/ops-thirdparty/$f" "$d/" 2>/dev/null || true
  done
  for f in makeself json; do
    [ -e "$d/$f" ] || cp -a "/opt/step5/ascend-thirdparty/$f" "$d/" 2>/dev/null || true
  done
done

# build.sh insists on dos2unix and pigz>=2.4; the image carries neither.
mkdir -p /w/.stage/shim
cat > /w/.stage/shim/dos2unix <<'SH'
#!/bin/sh
rc=0
for f in "$@"; do
  case "$f" in -*) continue ;; esac
  [ -f "$f" ] || { echo "dos2unix: $f: No such file" >&2; rc=1; continue; }
  sed -i 's/\r$//' "$f"
done
exit $rc
SH
cat > /w/.stage/shim/pigz <<'SH'
#!/bin/sh
case "$1" in --version|-V|-h|--help) echo "pigz 2.8"; exit 0 ;; esac
exec gzip "$@"
SH
chmod +x /w/.stage/shim/*
export PATH=/w/.stage/shim:/usr/local/bin:$PATH

cd /w/ops-transformer
bash build.sh --ops=generic_block_sparse_attention,generic_block_sparse_attention_metadata \
  --soc=ascend950 --vendor_name=union_test --cann_3rd_lib_path=/w/.stage/transformer \
  -j"$JOBS" > /w/.stage/transformer.log 2>&1

cd /w/ops-nn
bash build.sh --pkg --experimental --ops=union_sparse_meta --soc=ascend950 \
  --vendor_name=custom_nn --cann_3rd_lib_path=/w/.stage/nn -j"$JOBS" > /w/.stage/nn-pkg.log 2>&1
bash build.sh --torch_extension --experimental --ops=union_sparse_meta --soc=ascend950 \
  --vendor_name=custom_nn --cann_3rd_lib_path=/w/.stage/nn -j"$JOBS" > /w/.stage/nn-whl.log 2>&1
INNER

echo "==> compiling in $IMAGE"
"$DOCKER" run --rm \
  -v "$HERE:/w" -v /usr/bin/pigz:/usr/bin/pigz:ro \
  -e JOBS="$JOBS" --entrypoint bash "$IMAGE" /w/.stage/in-container.sh

echo "==> flattening into prebuilt/"
rm -rf "$HERE/prebuilt"
mkdir -p "$HERE/prebuilt"
# CPack cannot always produce the .run shell (it needs makeself, which is not
# reachable offline); the staging payload is the same tree the .run expands to.
TR_STAGE="$HERE/ops-transformer/build/_CPack_Packages/Linux/External/cann-ops-transformer-union_test_linux-aarch64.run/packages/vendors/union_test_transformer"
[[ -d "$TR_STAGE" ]] || { echo "no CPack payload at $TR_STAGE" >&2; exit 1; }
cp -a "$TR_STAGE" "$HERE/prebuilt/union_test_transformer"

NN_RUN="$HERE/ops-nn/build_out/cann-ops-nn-custom_nn_linux-aarch64.run"
"$DOCKER" run --rm -v "$HERE:/w" --entrypoint bash "$IMAGE" -lc "
set -e
bash /w/ops-nn/build_out/cann-ops-nn-custom_nn_linux-aarch64.run \
  --install-path=/w/.stage/nn-install/custom_nn_nn -q --nox11 > /w/.stage/nn-install.log 2>&1
mv /w/.stage/nn-install/custom_nn_nn/vendors/custom_nn_nn /w/prebuilt/custom_nn_nn
rm -rf /w/.stage/nn-install
" || { echo "installing $NN_RUN failed; see .stage/nn-install.log" >&2; exit 1; }
cp "$HERE"/ops-nn/build_out/cann_ops_nn_custom_nn-*.whl "$HERE/prebuilt/" 2>/dev/null || true

# A vendor tree without op_api/lib installs nothing: the .run adds a second
# vendors/<name> level that python cannot see through.
for v in union_test_transformer custom_nn_nn; do
  [[ -d "$HERE/prebuilt/$v/op_api/lib" ]] || { echo "prebuilt/$v is not flattened" >&2; exit 1; }
done
grep -qs UnionSparseMeta "$HERE/prebuilt/custom_nn_nn/op_api/lib/libcust_opapi.so" \
  || { echo "custom_nn_nn does not export UnionSparseMeta" >&2; exit 1; }

((CLEAN)) && rm -rf "$HERE/.stage"
echo "==> prebuilt/ ready:"
du -sh "$HERE"/prebuilt/*
