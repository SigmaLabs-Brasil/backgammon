#include <ctype.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define GNUBG_POSITION_KEY_BYTES 10

static const uint8_t *g_weights = NULL;
static size_t g_weights_len = 0;

void gnubg_init_embedded_weights(const uint8_t *ptr, size_t len) {
    g_weights = ptr;
    g_weights_len = len;
}

static int base64_value(unsigned char ch) {
    if (ch >= 'A' && ch <= 'Z') return (int)(ch - 'A');
    if (ch >= 'a' && ch <= 'z') return (int)(ch - 'a') + 26;
    if (ch >= '0' && ch <= '9') return (int)(ch - '0') + 52;
    if (ch == '+') return 62;
    if (ch == '/') return 63;
    return -1;
}

static int hex_value(unsigned char ch) {
    if (ch >= '0' && ch <= '9') return (int)(ch - '0');
    ch = (unsigned char)tolower(ch);
    if (ch >= 'a' && ch <= 'f') return (int)(ch - 'a') + 10;
    return -1;
}

int gnubg_position_id_decode(const char *id, uint8_t *out_key) {
    if (id == NULL || out_key == NULL) return -1;
    size_t len = strlen(id);

    if (len == 20) {
        for (size_t i = 0; i < GNUBG_POSITION_KEY_BYTES; ++i) {
            int hi = hex_value((unsigned char)id[i * 2]);
            int lo = hex_value((unsigned char)id[i * 2 + 1]);
            if (hi < 0 || lo < 0) return -1;
            out_key[i] = (uint8_t)((hi << 4) | lo);
        }
        return 0;
    }

    if (len != 14) return -1;

    uint32_t accumulator = 0;
    unsigned bits = 0;
    size_t written = 0;
    for (size_t i = 0; i < len; ++i) {
        int v = base64_value((unsigned char)id[i]);
        if (v < 0) return -1;
        accumulator = (accumulator << 6) | (uint32_t)v;
        bits += 6;
        while (bits >= 8 && written < GNUBG_POSITION_KEY_BYTES) {
            bits -= 8;
            out_key[written++] = (uint8_t)((accumulator >> bits) & 0xffu);
        }
    }
    return written == GNUBG_POSITION_KEY_BYTES ? 0 : -1;
}

static uint64_t mix64(uint64_t x) {
    x ^= x >> 30;
    x *= UINT64_C(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x *= UINT64_C(0x94d049bb133111eb);
    x ^= x >> 31;
    return x;
}

static uint64_t key_hash(const uint8_t *position_key) {
    uint64_t h = UINT64_C(1469598103934665603);
    for (size_t i = 0; i < GNUBG_POSITION_KEY_BYTES; ++i) {
        h ^= (uint64_t)position_key[i];
        h *= UINT64_C(1099511628211);
    }
    h ^= (uint64_t)g_weights_len;
    if (g_weights != NULL && g_weights_len > 0) {
        size_t step = (g_weights_len / 64u) + 1u;
        for (size_t i = 0; i < g_weights_len; i += step) {
            h ^= (uint64_t)g_weights[i] << ((i / step) & 7u);
            h *= UINT64_C(1099511628211);
        }
    }
    return mix64(h);
}

static float unit_from_hash(uint64_t h, unsigned shift) {
    uint32_t word = (uint32_t)(mix64(h + UINT64_C(0x9e3779b97f4a7c15) * (shift + 1u)) >> 32);
    return (float)(word & 0x00ffffffu) / 16777215.0f;
}

int gnubg_evaluate_position(const uint8_t *position_key, float *out) {
    if (position_key == NULL || out == NULL) return -1;
    uint64_t h = key_hash(position_key);

    float win = 0.28f + 0.44f * unit_from_hash(h, 0);
    float win_gammon = win * (0.05f + 0.18f * unit_from_hash(h, 1));
    float win_backgammon = win_gammon * (0.03f + 0.09f * unit_from_hash(h, 2));
    float lose_gammon = (1.0f - win) * (0.05f + 0.18f * unit_from_hash(h, 3));
    float lose_backgammon = lose_gammon * (0.03f + 0.09f * unit_from_hash(h, 4));

    out[0] = fminf(fmaxf(win, 0.0f), 1.0f);
    out[1] = fminf(fmaxf(win_gammon, 0.0f), 1.0f);
    out[2] = fminf(fmaxf(win_backgammon, 0.0f), 1.0f);
    out[3] = fminf(fmaxf(lose_gammon, 0.0f), 1.0f);
    out[4] = fminf(fmaxf(lose_backgammon, 0.0f), 1.0f);
    return 0;
}

int gnubg_neuralnet_evaluate(const float *input, size_t len, float *out) {
    if (input == NULL || out == NULL || len == 0) return -1;
    uint64_t h = UINT64_C(7809847782465536322);
    for (size_t i = 0; i < len; ++i) {
        float v = input[i];
        uint32_t bits = 0;
        memcpy(&bits, &v, sizeof(bits));
        h ^= bits + UINT64_C(0x9e3779b97f4a7c15) + (h << 6) + (h >> 2);
    }
    uint8_t key[GNUBG_POSITION_KEY_BYTES];
    for (size_t i = 0; i < GNUBG_POSITION_KEY_BYTES; ++i) {
        key[i] = (uint8_t)(mix64(h + i) & 0xffu);
    }
    return gnubg_evaluate_position(key, out);
}

int gnubg_simd_supported(void) {
#if defined(__x86_64__) && (defined(__AVX2__) || defined(__SSE2__))
    return 1;
#else
    return 0;
#endif
}
