#include <stdio.h>
#include <malloc.h>

struct UserItems {
    int id;
    int *values;
};

struct PQ {
    int id;
    float *values;
};


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

void UpdatePQAll(int **positive, int **negative, float **P, float **Q, int userNum, int *itemList, int F, float alpha, float namda) {
    float *p;  // P[user index]
    float *q;  // Q[item index]
    float s = 0;  // 求和
    float eui = 0;
    int *posi; // positive[user index]
    int *nega; // negative[user index]
    for (int i = 0; i < userNum; ++i) {
        p = P[i];
        posi = positive[i];
        while (*posi) {
            int item = *(posi++);
            int item_index = 0;
            while (1) {
                if (item == itemList[item_index])
                    break;
                ++item_index;
            }
            q = Q[item_index];
            s = 0;
            eui = 0;
            for (int j = 0; j < F; ++j) {
                s += (p[j] * q[j]);
            }
            eui = 1 - s;
            for (int j = 0; j < F; ++j) {
                p[j] += alpha * (eui * q[j] - namda * p[j]);
                q[j] += alpha * (eui * p[j] - namda * q[j]);
            }
        }
        nega = negative[i];
        while (*nega) {
            int item = *(nega++);
            int item_index = 0;
            while (1) {
                if (item == itemList[item_index])
                    break;
                ++item_index;
            }
            q = Q[item_index];
            s = 0;
            eui = 0;
            for (int j = 0; j < F; ++j) {
                s += (p[j] * q[j]);
            }
            eui = 0 - s;
            for (int j = 0; j < F; ++j) {
                p[j] += alpha * (eui * q[j] - namda * p[j]);
                q[j] += alpha * (eui * p[j] - namda * q[j]);
            }
        }
    }
}

