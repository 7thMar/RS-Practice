#include <stdio.h>
#include <malloc.h>

void UpdatePQPositive(float *P, float *Q, int F, float alpha, float namda) {
    float s = 0;
    float eui = 0;
    for (int i = 0; i < F; ++i) {
        s += (P[i] * Q[i]);
    }
    eui = 1 - s;
    for (int i = 0; i < F; ++i) {
        P[i] += alpha * (eui * Q[i] - namda * P[i]);
        Q[i] += alpha * (eui * P[i] - namda * Q[i]);
    }
}
void UpdatePQNegative(float *P, float *Q, int F, float alpha, float namda) {
    float s = 0;
    float eui = 0;
    for (int i = 0; i < F; ++i) {
        s += (P[i] * Q[i]);
    }
    eui = 0 - s;
    for (int i = 0; i < F; ++i) {
        P[i] += alpha * (eui * Q[i] - namda * P[i]);
        Q[i] += alpha * (eui * P[i] - namda * Q[i]);
    }
}
int SelectNegativeSample(int *items_pool, int *items_index, int *positive, int *negative, int sample_length) {
    int item = 0; // 当前挑选的item
    int num = 0; // 挑选的数目
    int n = 0; //
    for (int i = 0; i < 3 * sample_length; ++i) {
        item = items_pool[items_index[i]];
        n = 0;
        while (positive[n]) {
            if (positive[n] == item) {
                break;
            }
            ++n;
        }
        if (positive[n]) {
            continue;
        }
        n = 0;
        while (negative[n]) {
            if (negative[n] == item) {
                break;
            }
            ++n;
        }
        if (negative[n]) {
            continue;
        }
        negative[num++] = item;
        if (num == sample_length) {
            break;
        }
    }
    return num;
}
