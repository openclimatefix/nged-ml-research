/* Viterbi pass of experiment 734 in C, for e00734_native.py.
 *
 * Same arithmetic as `e00734_hmm.viterbi`: float32 additions only (no fused multiply-add and no
 * fast-math), strict `>` so that the first maximum wins like `numpy.argmax`, and the same factored
 * steps (offset and move state, then permanent level, then gain), each storing its argmax.
 *
 * Layout of every state array: (om, l, g) row-major, with NOM offset-and-move rows, NL permanent
 * levels, and NG gains. The columns `c = l * NG + g` of one row are contiguous, so the loops over
 * `c` vectorise.
 */
#include <stdint.h>
#include <string.h>

#define NOM 123
#define NL 21
#define NG 5
#define NC (NL * NG)
#define NO 41

/* Run blocks `0 .. n - 1` of a chunk. `delta` holds the scores of the block before the chunk and
 * is updated in place. When `first_has_prev` is 0 the first block has no transition (block 0 of
 * the record). `le` is the float32 log emission of every block, shape (n, NO, NC).
 * `log_t` holds `n_t` matrices `[j][i]`, the log probability of moving from row i to row j, and
 * `t_index` names the matrix of each block. `log_l[to][from]` and `log_g[to][from]` are the log
 * transitions of the permanent level and the gain. The three backpointer arrays are
 * (n, NOM, NC) and hold the argmax of each step. */
void viterbi_chunk(
    int n,
    int first_has_prev,
    const float *le,
    const float *log_t,
    const int32_t *t_index,
    const float *log_l,
    const float *log_g,
    float *delta,
    int8_t *bp_om,
    int8_t *bp_l,
    int8_t *bp_g)
{
    static float d1[NOM * NC];
    static float d2[NOM * NC];
    static float best[NC];
    static int32_t arg[NC];
    for (int t = 0; t < n; t++) {
        if (t > 0 || first_has_prev) {
            const float *lt = log_t + (size_t)t_index[t] * NOM * NOM;
            int8_t *bo = bp_om + (size_t)t * NOM * NC;
            for (int j = 0; j < NOM; j++) {
                const float *row = lt + (size_t)j * NOM;
                for (int c = 0; c < NC; c++) {
                    best[c] = delta[c] + row[0];
                    arg[c] = 0;
                }
                for (int i = 1; i < NOM; i++) {
                    const float w = row[i];
                    const float *dr = delta + (size_t)i * NC;
                    for (int c = 0; c < NC; c++) {
                        const float s = dr[c] + w;
                        const int better = s > best[c];
                        best[c] = better ? s : best[c];
                        arg[c] = better ? i : arg[c];
                    }
                }
                for (int c = 0; c < NC; c++) {
                    d1[(size_t)j * NC + c] = best[c];
                    bo[(size_t)j * NC + c] = (int8_t)arg[c];
                }
            }
            int8_t *bl = bp_l + (size_t)t * NOM * NC;
            for (int j = 0; j < NOM; j++) {
                const float *src = d1 + (size_t)j * NC;
                for (int to = 0; to < NL; to++) {
                    const float *w = log_l + to * NL;
                    float bst[NG];
                    int32_t ag[NG];
                    for (int g = 0; g < NG; g++) {
                        bst[g] = src[g] + w[0];
                        ag[g] = 0;
                    }
                    for (int from = 1; from < NL; from++) {
                        for (int g = 0; g < NG; g++) {
                            const float s = src[from * NG + g] + w[from];
                            const int better = s > bst[g];
                            bst[g] = better ? s : bst[g];
                            ag[g] = better ? from : ag[g];
                        }
                    }
                    for (int g = 0; g < NG; g++) {
                        d2[(size_t)j * NC + to * NG + g] = bst[g];
                        bl[(size_t)j * NC + to * NG + g] = (int8_t)ag[g];
                    }
                }
            }
            int8_t *bg = bp_g + (size_t)t * NOM * NC;
            for (int j = 0; j < NOM; j++) {
                for (int l = 0; l < NL; l++) {
                    const float *src = d2 + (size_t)j * NC + l * NG;
                    for (int to = 0; to < NG; to++) {
                        const float *w = log_g + to * NG;
                        float bst = src[0] + w[0];
                        int32_t ag = 0;
                        for (int from = 1; from < NG; from++) {
                            const float s = src[from] + w[from];
                            if (s > bst) {
                                bst = s;
                                ag = from;
                            }
                        }
                        delta[(size_t)j * NC + l * NG + to] = bst;
                        bg[(size_t)j * NC + l * NG + to] = (int8_t)ag;
                    }
                }
            }
        }
        const float *e = le + (size_t)t * NO * NC;
        float top = -__builtin_inff();
        for (int j = 0; j < NOM; j++) {
            const float *er = e + (size_t)(j % NO) * NC;
            float *dr = delta + (size_t)j * NC;
            for (int c = 0; c < NC; c++) {
                dr[c] = dr[c] + er[c];
                top = dr[c] > top ? dr[c] : top;
            }
        }
        for (int k = 0; k < NOM * NC; k++) {
            delta[k] = delta[k] - top;
        }
    }
}
