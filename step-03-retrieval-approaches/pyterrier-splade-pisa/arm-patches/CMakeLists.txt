cmake_minimum_required(VERSION 3.15)

project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} -mtune=generic")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mtune=generic")


set(PISA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(PISA_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(PISA_ENABLE_BENCHMARKING OFF CACHE BOOL "" FORCE)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module NumPy)
include_directories(${Python_NumPy_INCLUDE_DIRS})



# Write file only if content changed.
function(_pisa_write_if_changed _path _content)
    set(_existing "")
    if(EXISTS "${_path}")
        file(READ "${_path}" _existing)
    endif()
    if(NOT _existing STREQUAL "${_content}")
        file(WRITE "${_path}" "${_content}")
        message(STATUS "pyterrier_pisa: wrote ${_path}")
    endif()
endfunction()

# Wrap x86intrin include in architecture guard.
function(_pisa_guard_x86intrin _file)
    if(NOT EXISTS "${_file}")
        return()
    endif()
    file(READ "${_file}" _content)
    string(FIND "${_content}" "#include <x86intrin.h>" _inc_pos)
    if(_inc_pos LESS 0)
        return()
    endif()
    string(SUBSTRING "${_content}" 0 ${_inc_pos} _before)
    if(_before MATCHES "defined\\(__x86_64__\\)")
        return()
    endif()
    string(REGEX REPLACE
        "#include <x86intrin\\.h>"
        "#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)\n#include <x86intrin.h>\n#endif"
        _content "${_content}")
    file(WRITE "${_file}" "${_content}")
    message(STATUS "pyterrier_pisa: ARM guard applied to ${_file}")
endfunction()

# Strip -march=native from submodule CMakeLists.
function(_pisa_strip_march_native _file)
    if(NOT EXISTS "${_file}")
        return()
    endif()
    file(READ "${_file}" _content)
    string(REGEX REPLACE
        "set\\(CMAKE_(C|CXX)_FLAGS \"\\\$\\{CMAKE_(C|CXX)_FLAGS\\} -march=native\"\\)"
        "# (-march=native removed; arch flags set by parent CMakeLists)"
        _new "${_content}")
    if(NOT _new STREQUAL "${_content}")
        file(WRITE "${_file}" "${_new}")
        message(STATUS "pyterrier_pisa: stripped -march=native from ${_file}")
    endif()
endfunction()

# Apply x86intrin guards.
_pisa_guard_x86intrin("${CMAKE_CURRENT_SOURCE_DIR}/pisa/include/pisa/util/intrinsics.hpp")
_pisa_guard_x86intrin("${CMAKE_CURRENT_SOURCE_DIR}/pisa/include/pisa/codec/VarIntG8IU.h")

# Strip -march=native.
_pisa_strip_march_native("${CMAKE_CURRENT_SOURCE_DIR}/pisa/CMakeLists.txt")
_pisa_strip_march_native("${CMAKE_CURRENT_SOURCE_DIR}/pisa/external/CMakeLists.txt")


set(_rgb_file "${CMAKE_CURRENT_SOURCE_DIR}/pisa/include/pisa/recursive_graph_bisection.hpp")
if(EXISTS "${_rgb_file}")
    file(READ "${_rgb_file}" _rgb_content)
    if(_rgb_content MATCHES "__m128 _deg")
        # Replace SSE expb() with scalar version.
        string(REGEX REPLACE
            "__m128 _deg[^@]*return a\\[3\\] - a\\[2\\] \\+ a\\[1\\] - a\\[0\\];"
            "/* ARM patch: scalar expb(). */ return double(deg1) * logn1 - double(deg1) * log2(deg1 + 1) + double(deg2) * logn2 - double(deg2) * log2(deg2 + 1);"
            _rgb_new "${_rgb_content}")
        if(NOT _rgb_new STREQUAL _rgb_content)
            file(WRITE "${_rgb_file}" "${_rgb_new}")
            message(STATUS "pyterrier_pisa: scalar expb() applied to ${_rgb_file}")
        endif()
    endif()
endif()


# MaskedVByte: scalar replacement for SSE/AVX.
  set(_mb "${CMAKE_CURRENT_SOURCE_DIR}/pisa/external/MaskedVByte")
  if(IS_DIRECTORY "${_mb}")
    _pisa_write_if_changed("${_mb}/src/varintdecode.c" [=[
/* ARM stub: scalar VByte. */
#include <stdint.h>
#include <stddef.h>
#include "../include/varintdecode.h"
size_t masked_vbyte_decode(const uint8_t *in, uint32_t *out, uint64_t length) {
    size_t pos = 0;
    for (uint64_t i = 0; i < length; i++) {
        uint32_t v = 0; int s = 0;
        while (in[pos] & 0x80) { v |= (uint32_t)(in[pos++] & 0x7F) << s; s += 7; }
        out[i] = v | ((uint32_t)in[pos++] << s);
    }
    return pos;
}
size_t masked_vbyte_decode_delta(const uint8_t *in, uint32_t *out, uint64_t length, uint32_t prev) {
    size_t pos = 0;
    for (uint64_t i = 0; i < length; i++) {
        uint32_t v = 0; int s = 0;
        while (in[pos] & 0x80) { v |= (uint32_t)(in[pos++] & 0x7F) << s; s += 7; }
        v |= (uint32_t)in[pos++] << s;
        prev += v; out[i] = prev;
    }
    return pos;
}
size_t masked_vbyte_decode_fromcompressedsize(const uint8_t *in, uint32_t *out, size_t inputsize) {
    size_t pos = 0, n = 0;
    while (pos < inputsize) {
        uint32_t v = 0; int s = 0;
        while (in[pos] & 0x80) { v |= (uint32_t)(in[pos++] & 0x7F) << s; s += 7; }
        out[n++] = v | ((uint32_t)in[pos++] << s);
    }
    return n;
}
size_t masked_vbyte_decode_fromcompressedsize_delta(const uint8_t *in, uint32_t *out, size_t inputsize, uint32_t prev) {
    size_t pos = 0, n = 0;
    while (pos < inputsize) {
        uint32_t v = 0; int s = 0;
        while (in[pos] & 0x80) { v |= (uint32_t)(in[pos++] & 0x7F) << s; s += 7; }
        v |= (uint32_t)in[pos++] << s;
        prev += v; out[n++] = prev;
    }
    return n;
}
uint32_t masked_vbyte_select_delta(const uint8_t *in, uint64_t length, uint32_t prev, size_t slot) {
    size_t pos = 0;
    for (uint64_t i = 0; i <= slot && i < length; i++) {
        uint32_t v = 0; int s = 0;
        while (in[pos] & 0x80) { v |= (uint32_t)(in[pos++] & 0x7F) << s; s += 7; }
        v |= (uint32_t)in[pos++] << s;
        prev += v;
    }
    return prev;
}
int masked_vbyte_search_delta(const uint8_t *in, uint64_t length, uint32_t prev, uint32_t key, uint32_t *presult) {
    size_t pos = 0;
    for (uint64_t i = 0; i < length; i++) {
        uint32_t v = 0; int s = 0;
        while (in[pos] & 0x80) { v |= (uint32_t)(in[pos++] & 0x7F) << s; s += 7; }
        v |= (uint32_t)in[pos++] << s;
        prev += v;
        if (prev >= key) { *presult = prev; return (int)i; }
    }
    return -1;
}
]=])
    _pisa_write_if_changed("${_mb}/src/varintencode.c" [=[
/* ARM stub: scalar VByte. */
#include <stdint.h>
#include <stddef.h>
#include "../include/varintencode.h"
size_t vbyte_encode(uint32_t *in, size_t length, uint8_t *bout) {
    size_t pos = 0;
    for (size_t i = 0; i < length; i++) {
        uint32_t v = in[i];
        while (v > 0x7F) { bout[pos++] = (uint8_t)((v & 0x7F) | 0x80); v >>= 7; }
        bout[pos++] = (uint8_t)v;
    }
    return pos;
}
size_t vbyte_encode_delta(uint32_t *in, size_t length, uint8_t *bout, uint32_t prev) {
    size_t pos = 0;
    for (size_t i = 0; i < length; i++) {
        uint32_t v = in[i] - prev; prev = in[i];
        while (v > 0x7F) { bout[pos++] = (uint8_t)((v & 0x7F) | 0x80); v >>= 7; }
        bout[pos++] = (uint8_t)v;
    }
    return pos;
}
]=])
  endif()

  # simdcomp: empty stubs.
  set(_sc "${CMAKE_CURRENT_SOURCE_DIR}/pisa/external/simdcomp")
  if(IS_DIRECTORY "${_sc}")
    _pisa_write_if_changed("${_sc}/src/simdbitpacking.c" "/* ARM stub. */\n")
    _pisa_write_if_changed("${_sc}/src/simdcomputil.c" "/* ARM stub. */\n")
  endif()

  # streamvbyte: empty stubs.
  set(_svb "${CMAKE_CURRENT_SOURCE_DIR}/pisa/external/streamvbyte")
  if(IS_DIRECTORY "${_svb}")
    _pisa_write_if_changed("${_svb}/src/streamvbyte.c" "/* ARM stub. */\n")
    _pisa_write_if_changed("${_svb}/src/streamvbytedelta.c" "/* ARM stub. */\n")
  endif()

  # FastPFor: minimal CMakeLists with empty TU.
  set(_fp "${CMAKE_CURRENT_SOURCE_DIR}/pisa/external/FastPFor")
  if(IS_DIRECTORY "${_fp}")
    _pisa_write_if_changed("${_fp}/fastpfor_arm_stub.cpp"
        "/* ARM stub: empty TU. */\n")
    _pisa_write_if_changed("${_fp}/CMakeLists.txt" [=[
# ARM stub: minimal FastPFor build.
cmake_minimum_required(VERSION 3.5)
project(FastPFor CXX)
add_library(FastPFor STATIC fastpfor_arm_stub.cpp)
target_include_directories(FastPFor PUBLIC headers)
]=])
  endif()

  # QMX: empty INTERFACE library.
  set(_qmx_dir "${CMAKE_CURRENT_SOURCE_DIR}/pisa/external/QMX")
  if(IS_DIRECTORY "${_qmx_dir}")
    _pisa_write_if_changed("${_qmx_dir}/CMakeLists.txt"
        "# ARM stub.\nadd_library(QMX INTERFACE)\n")
  endif()

  # VarIntG8IU.h: empty placeholder class.
  _pisa_write_if_changed(
      "${CMAKE_CURRENT_SOURCE_DIR}/pisa/include/pisa/codec/VarIntG8IU.h"
      [=[/* ARM stub: empty placeholder. */
#pragma once
#include <cstdint>
#include <vector>
namespace pisa {
class VarIntG8IU {
  public:
    VarIntG8IU() = default;
    ~VarIntG8IU() = default;
};
}  // namespace pisa
]=])

  # Codec headers: route through interpolative_block.
  set(_arm_codec_stubs
      "qmx:qmx_block"
      "simple8b:simple8b_block"
      "simple16:simple16_block"
      "streamvbyte:streamvbyte_block"
  )
  foreach(_spec IN LISTS _arm_codec_stubs)
    string(REGEX MATCHALL "[^:]+" _parts "${_spec}")
    list(GET _parts 0 _hdr)
    list(GET _parts 1 _struct)
    set(_target "${CMAKE_CURRENT_SOURCE_DIR}/pisa/include/pisa/codec/${_hdr}.hpp")
    set(_body
"/* ARM stub: routed through interpolative_block. */
#pragma once

#include \"codec/block_codecs.hpp\"
#include <cstdint>
#include <vector>

namespace pisa {
struct ${_struct} {
    static const uint64_t block_size = 128;
    static void encode(uint32_t const* in, uint32_t sum_of_values, size_t n, std::vector<uint8_t>& out)
    {
        interpolative_block::encode(in, sum_of_values, n, out);
    }
    static uint8_t const* decode(uint8_t const* in, uint32_t* out, uint32_t sum_of_values, size_t n)
    {
        return interpolative_block::decode(in, out, sum_of_values, n);
    }
};
}  // namespace pisa
")
    _pisa_write_if_changed("${_target}" "${_body}")
  endforeach()

  # varintgb.hpp: keep VarIntGB, drop FastPFor include.
  _pisa_write_if_changed(
      "${CMAKE_CURRENT_SOURCE_DIR}/pisa/include/pisa/codec/varintgb.hpp"
      [=[/* ARM stub: FastPFor include removed; varintgb_block routed through interpolative_block. */
#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "codec/block_codecs.hpp"

namespace pisa {

template <bool delta = false>
class VarIntGB {
  public:
    size_t encodeArray(const uint32_t* in, const size_t length, uint8_t* out)
    {
        uint32_t prev = 0;  // for delta
        const uint8_t* const initbout = out;

        size_t k = 0;
        for (; k + 3 < length; k += 4) {
            uint8_t* keyp = out++;
            *keyp = 0;
            {
                const uint32_t val = delta ? in[k] - prev : in[k];
                if (delta) { prev = in[k]; }
                if (val < (1U << 8)) {
                    *out++ = static_cast<uint8_t>(val);
                } else if (val < (1U << 16)) {
                    *out++ = static_cast<uint8_t>(val);
                    *out++ = static_cast<uint8_t>(val >> 8);
                    *keyp = static_cast<uint8_t>(1);
                } else if (val < (1U << 24)) {
                    *out++ = static_cast<uint8_t>(val);
                    *out++ = static_cast<uint8_t>(val >> 8);
                    *out++ = static_cast<uint8_t>(val >> 16);
                    *keyp = static_cast<uint8_t>(2);
                } else {
                    *reinterpret_cast<uint32_t*>(out) = val;
                    out += 4;
                    *keyp = static_cast<uint8_t>(3);
                }
            }
            {
                const uint32_t val = delta ? in[k + 1] - prev : in[k + 1];
                if (delta) { prev = in[k + 1]; }
                if (val < (1U << 8)) {
                    *out++ = static_cast<uint8_t>(val);
                } else if (val < (1U << 16)) {
                    *out++ = static_cast<uint8_t>(val);
                    *out++ = static_cast<uint8_t>(val >> 8);
                    *keyp |= static_cast<uint8_t>(1 << 2);
                } else if (val < (1U << 24)) {
                    *out++ = static_cast<uint8_t>(val);
                    *out++ = static_cast<uint8_t>(val >> 8);
                    *out++ = static_cast<uint8_t>(val >> 16);
                    *keyp |= static_cast<uint8_t>(2 << 2);
                } else {
                    *reinterpret_cast<uint32_t*>(out) = val;
                    out += 4;
                    *keyp |= static_cast<uint8_t>(3 << 2);
                }
            }
            {
                const uint32_t val = delta ? in[k + 2] - prev : in[k + 2];
                if (delta) { prev = in[k + 2]; }
                if (val < (1U << 8)) {
                    *out++ = static_cast<uint8_t>(val);
                } else if (val < (1U << 16)) {
                    *out++ = static_cast<uint8_t>(val);
                    *out++ = static_cast<uint8_t>(val >> 8);
                    *keyp |= static_cast<uint8_t>(1 << 4);
                } else if (val < (1U << 24)) {
                    *out++ = static_cast<uint8_t>(val);
                    *out++ = static_cast<uint8_t>(val >> 8);
                    *out++ = static_cast<uint8_t>(val >> 16);
                    *keyp |= static_cast<uint8_t>(2 << 4);
                } else {
                    *reinterpret_cast<uint32_t*>(out) = val;
                    out += 4;
                    *keyp |= static_cast<uint8_t>(3 << 4);
                }
            }
            {
                const uint32_t val = delta ? in[k + 3] - prev : in[k + 3];
                if (delta) { prev = in[k + 3]; }
                if (val < (1U << 8)) {
                    *out++ = static_cast<uint8_t>(val);
                } else if (val < (1U << 16)) {
                    *out++ = static_cast<uint8_t>(val);
                    *out++ = static_cast<uint8_t>(val >> 8);
                    *keyp |= static_cast<uint8_t>(1 << 6);
                } else if (val < (1U << 24)) {
                    *out++ = static_cast<uint8_t>(val);
                    *out++ = static_cast<uint8_t>(val >> 8);
                    *out++ = static_cast<uint8_t>(val >> 16);
                    *keyp |= static_cast<uint8_t>(2 << 6);
                } else {
                    *reinterpret_cast<uint32_t*>(out) = val;
                    out += 4;
                    *keyp |= static_cast<uint8_t>(3 << 6);
                }
            }
        }

        if (k < length) {
            uint8_t* keyp = out++;
            *keyp = 0;
            for (int j = 0; k < length && j < 8; j += 2, ++k) {
                const uint32_t val = delta ? in[k] - prev : in[k];
                if (delta) { prev = in[k]; }
                if (val < (1U << 8)) {
                    *out++ = static_cast<uint8_t>(val);
                } else if (val < (1U << 16)) {
                    *out++ = static_cast<uint8_t>(val);
                    *out++ = static_cast<uint8_t>(val >> 8);
                    *keyp |= static_cast<uint8_t>(1 << j);
                } else if (val < (1U << 24)) {
                    *out++ = static_cast<uint8_t>(val);
                    *out++ = static_cast<uint8_t>(val >> 8);
                    *out++ = static_cast<uint8_t>(val >> 16);
                    *keyp |= static_cast<uint8_t>(2 << j);
                } else {
                    *reinterpret_cast<uint32_t*>(out) = val;
                    out += 4;
                    *keyp |= static_cast<uint8_t>(3 << j);
                }
            }
        }
        const size_t storageinbytes = out - initbout;
        return storageinbytes;
    }

    size_t decodeArray(const uint8_t* in, const size_t n, uint32_t* out)
    {
        uint32_t prev = 0;  // for delta
        const uint8_t* initin = in;
        uint32_t val;
        size_t k = 0;
        while (k + 3 < n) {
            in = delta ? decodeGroupVarIntDelta(in, &prev, out) : decodeGroupVarInt(in, out);
            out += 4;
            k += 4;
        }
        while (k < n) {
            uint8_t key = *in++;
            for (int j = 0; k < n && j < 4; j++) {
                const uint32_t howmanybyte = key & 3;
                key = static_cast<uint8_t>(key >> 2);
                val = static_cast<uint32_t>(*in++);
                if (howmanybyte >= 1) {
                    val |= (static_cast<uint32_t>(*in++) << 8);
                    if (howmanybyte >= 2) {
                        val |= (static_cast<uint32_t>(*in++) << 16);
                        if (howmanybyte >= 3) {
                            val |= (static_cast<uint32_t>(*in++) << 24);
                        }
                    }
                }
                prev = (delta ? prev : 0) + val;
                *out++ = prev;
                k++;
            }
        }
        return in - initin;
    }

  protected:
    static uint32_t mask[4];

    const uint8_t* decodeGroupVarInt(const uint8_t* in, uint32_t* out)
    {
        const uint32_t sel = *in++;
        if (sel == 0) {
            out[0] = static_cast<uint32_t>(in[0]);
            out[1] = static_cast<uint32_t>(in[1]);
            out[2] = static_cast<uint32_t>(in[2]);
            out[3] = static_cast<uint32_t>(in[3]);
            return in + 4;
        }
        const uint32_t sel1 = (sel & 3);
        *out++ = *(reinterpret_cast<const uint32_t*>(in)) & mask[sel1];
        in += sel1 + 1;
        const uint32_t sel2 = ((sel >> 2) & 3);
        *out++ = *(reinterpret_cast<const uint32_t*>(in)) & mask[sel2];
        in += sel2 + 1;
        const uint32_t sel3 = ((sel >> 4) & 3);
        *out++ = *(reinterpret_cast<const uint32_t*>(in)) & mask[sel3];
        in += sel3 + 1;
        const uint32_t sel4 = (sel >> 6);
        *out++ = *(reinterpret_cast<const uint32_t*>(in)) & mask[sel4];
        in += sel4 + 1;
        return in;
    }

    const uint8_t* decodeGroupVarIntDelta(const uint8_t* in, uint32_t* val, uint32_t* out)
    {
        const uint32_t sel = *in++;
        if (sel == 0) {
            out[0] = (*val += static_cast<uint32_t>(in[0]));
            out[1] = (*val += static_cast<uint32_t>(in[1]));
            out[2] = (*val += static_cast<uint32_t>(in[2]));
            out[3] = (*val += static_cast<uint32_t>(in[3]));
            return in + 4;
        }
        const uint32_t sel1 = (sel & 3);
        *val += *(reinterpret_cast<const uint32_t*>(in)) & mask[sel1];
        *out++ = *val;
        in += sel1 + 1;
        const uint32_t sel2 = ((sel >> 2) & 3);
        *val += *(reinterpret_cast<const uint32_t*>(in)) & mask[sel2];
        *out++ = *val;
        in += sel2 + 1;
        const uint32_t sel3 = ((sel >> 4) & 3);
        *val += *(reinterpret_cast<const uint32_t*>(in)) & mask[sel3];
        *out++ = *val;
        in += sel3 + 1;
        const uint32_t sel4 = (sel >> 6);
        *val += *(reinterpret_cast<const uint32_t*>(in)) & mask[sel4];
        *out++ = *val;
        in += sel4 + 1;
        return in;
    }
};

template <bool delta>
uint32_t VarIntGB<delta>::mask[4] = {0xFF, 0xFFFF, 0xFFFFFF, 0xFFFFFFFF};

struct varintgb_block {
    static const uint64_t block_size = 128;
    static void encode(uint32_t const* in, uint32_t sum_of_values, size_t n, std::vector<uint8_t>& out)
    {
        interpolative_block::encode(in, sum_of_values, n, out);
    }
    static uint8_t const* decode(uint8_t const* in, uint32_t* out, uint32_t sum_of_values, size_t n)
    {
        return interpolative_block::decode(in, out, sum_of_values, n);
    }
};

}  // namespace pisa
]=])

  # simdbp.hpp: route through interpolative_block.
  _pisa_write_if_changed(
      "${CMAKE_CURRENT_SOURCE_DIR}/pisa/include/pisa/codec/simdbp.hpp"
      [=[/* ARM stub: routed through interpolative_block. */
#pragma once

#include "codec/block_codecs.hpp"
#include <cstdint>
#include <vector>

namespace pisa {
struct simdbp_block {
    static const uint64_t block_size = 128;
    static void encode(uint32_t const* in, uint32_t sum_of_values, size_t n, std::vector<uint8_t>& out)
    {
        interpolative_block::encode(in, sum_of_values, n, out);
    }
    static uint8_t const* decode(uint8_t const* in, uint32_t* out, uint32_t sum_of_values, size_t n)
    {
        return interpolative_block::decode(in, out, sum_of_values, n);
    }
};
}  // namespace pisa
]=])

  # block_codecs.hpp: SIMD-free rewrite.
  _pisa_write_if_changed(
      "${CMAKE_CURRENT_SOURCE_DIR}/pisa/include/pisa/codec/block_codecs.hpp"
      [=[/* ARM stub: SIMD-free block_codecs. */
#pragma once

#include <array>
#include <cassert>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include "codec/integer_codes.hpp"
#include "codec/interpolative_coding.hpp"
#include "util/compiler_attribute.hpp"
#include "util/likely.hpp"
#include "util/util.hpp"

namespace pisa {

// VByte encoder/decoder that tracks value count instead of buffer size.
class TightVariableByte {
  public:
    template <uint32_t i>
    static uint8_t extract7bits(const uint32_t val)
    {
        return static_cast<uint8_t>((val >> (7 * i)) & ((1U << 7) - 1));
    }

    template <uint32_t i>
    static uint8_t extract7bitsmaskless(const uint32_t val)
    {
        return static_cast<uint8_t>(val >> (7 * i));
    }

    static void encode(const uint32_t* in, const size_t length, uint8_t* out, size_t& nvalue)
    {
        uint8_t* bout = out;
        for (size_t k = 0; k < length; ++k) {
            const uint32_t val = in[k];
            if (val < (1U << 7)) {
                *bout = val | 0x80U;
                ++bout;
            } else if (val < (1U << 14)) {
                *bout = extract7bits<0>(val);
                ++bout;
                *bout = extract7bitsmaskless<1>(val) | 0x80U;
                ++bout;
            } else if (val < (1U << 21)) {
                *bout = extract7bits<0>(val);
                ++bout;
                *bout = extract7bits<1>(val);
                ++bout;
                *bout = extract7bitsmaskless<2>(val) | 0x80U;
                ++bout;
            } else if (val < (1U << 28)) {
                *bout = extract7bits<0>(val);
                ++bout;
                *bout = extract7bits<1>(val);
                ++bout;
                *bout = extract7bits<2>(val);
                ++bout;
                *bout = extract7bitsmaskless<3>(val) | 0x80U;
                ++bout;
            } else {
                *bout = extract7bits<0>(val);
                ++bout;
                *bout = extract7bits<1>(val);
                ++bout;
                *bout = extract7bits<2>(val);
                ++bout;
                *bout = extract7bits<3>(val);
                ++bout;
                *bout = extract7bitsmaskless<4>(val) | 0x80U;
                ++bout;
            }
        }
        nvalue = bout - out;
    }

    static void encode_single(uint32_t val, std::vector<uint8_t>& out)
    {
        uint8_t buf[5];
        size_t nvalue;
        encode(&val, 1, buf, nvalue);
        out.insert(out.end(), buf, buf + nvalue);
    }

    static uint8_t const* decode(const uint8_t* in, uint32_t* out, size_t n)
    {
        const uint8_t* inbyte = in;
        for (size_t i = 0; i < n; ++i) {
            unsigned int shift = 0;
            for (uint32_t v = 0;; shift += 7) {
                uint8_t c = *inbyte++;
                v += ((c & 127) << shift);
                if ((c & 128)) {
                    *out++ = v;
                    break;
                }
            }
        }
        return inbyte;
    }

    static void decode(const uint8_t* in, uint32_t* out, size_t len, size_t& n)
    {
        const uint8_t* inbyte = in;
        const uint8_t* end = in + len;
        n = 0;
        while (inbyte < end) {
            unsigned int shift = 0;
            for (uint32_t v = 0;; shift += 7) {
                uint8_t c = *inbyte++;
                v += ((c & 127) << shift);
                if ((c & 128)) {
                    *out++ = v;
                    ++n;
                    break;
                }
            }
        }
    }
};

struct interpolative_block {
    static constexpr std::uint64_t block_size = 128;

    static void encode(uint32_t const* in, uint32_t sum_of_values, size_t n, std::vector<uint8_t>& out)
    {
        assert(n <= block_size);
        thread_local std::array<std::uint32_t, block_size> inbuf{};
        thread_local std::vector<uint32_t> outbuf;
        inbuf[0] = *in;
        for (size_t i = 1; i < n; ++i) {
            inbuf[i] = inbuf[i - 1] + in[i];
        }

        if (sum_of_values == uint32_t(-1)) {
            sum_of_values = inbuf[n - 1];
            TightVariableByte::encode_single(sum_of_values, out);
        }

        bit_writer bw(outbuf);
        bw.write_interpolative(inbuf.data(), n - 1, 0, sum_of_values);
        auto const* bufptr = reinterpret_cast<uint8_t const*>(outbuf.data());
        out.insert(out.end(), bufptr, bufptr + ceil_div(bw.size(), 8));
    }

    static uint8_t const* PISA_NOINLINE
    decode(uint8_t const* in, uint32_t* out, uint32_t sum_of_values, size_t n)
    {
        assert(n <= block_size);
        if (sum_of_values == std::numeric_limits<std::uint32_t>::max()) {
            in = TightVariableByte::decode(in, &sum_of_values, 1);
        }

        out[n - 1] = sum_of_values;
        size_t read_interpolative = 0;
        if (n > 1) {
            bit_reader br(in);
            br.read_interpolative(out, n - 1, 0, sum_of_values);
            for (size_t i = n - 1; i > 0; --i) {
                out[i] -= out[i - 1];
            }
            read_interpolative = ceil_div(br.position(), 8);
        }

        return in + read_interpolative;
    }
};

// ARM stub: routed through interpolative_block.
struct optpfor_block {
    static const uint64_t block_size = 128;

    static void encode(
        uint32_t const* in,
        uint32_t sum_of_values,
        size_t n,
        std::vector<uint8_t>& out,
        uint8_t const* /*b*/ = nullptr)
    {
        interpolative_block::encode(in, sum_of_values, n, out);
    }

    static uint8_t const* PISA_NOINLINE
    decode(uint8_t const* in, uint32_t* out, uint32_t sum_of_values, size_t n)
    {
        return interpolative_block::decode(in, out, sum_of_values, n);
    }
};

// ARM stub: routed through interpolative_block.
struct varint_G8IU_block {
    static const uint64_t block_size = 128;

    static void encode(uint32_t const* in, uint32_t sum_of_values, size_t n, std::vector<uint8_t>& out)
    {
        interpolative_block::encode(in, sum_of_values, n, out);
    }

    static uint8_t const* decode(uint8_t const* in, uint32_t* out, uint32_t sum_of_values, size_t n)
    {
        return interpolative_block::decode(in, out, sum_of_values, n);
    }
};

}  // namespace pisa
]=])

add_subdirectory(pisa)

Python_add_library(_pisathon MODULE src/pyterrier_pisa/_pisathon.cpp WITH_SOABI)
target_link_libraries(_pisathon PRIVATE pisa)
target_include_directories(_pisathon PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/pisa/external/GSL/include
)

install(TARGETS _pisathon DESTINATION pyterrier_pisa)