Here is the translation of the provided CUDA code to OpenMP:

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define B ((int)32)
#define T ((int)32)
#define THREADS ((int)B*T)
#define CLIMBINGS 150000
#define ALPHABET 26
#define totalBigrams ((int)ALPHABET*ALPHABET)
#define CAP ((float)999999.0)

#define ENCRYPTED_T "tteohtedanisroudesereguwocubsoitoabbofeiaiutsdheeisatsarsturesuaastniersrotnesctrctxdiwmhcusyenorndasmhaipnnptmaeecspegdeislwoheoiymreeotbsspiatoanihrelhwctftrhpuunhoianunreetrioettatlsnehtbaecpvgtltcirottonesnobeeeireaymrtohaawnwtesssvassirsrhabapnsynntitsittchitoosbtelmlaouitrehhwfeiaandeitciegfreoridhdcsheucrnoihdeoswobaceeaorgndlstigeearsotoetduedininttpedststntefoeaheoesuetvmmiorftuuhsurof"
#define ENCRYPTEDLEN ((int)sizeof(ENCRYPTED_T)-1)

#define DECRYPTED_T "thedistinctionbetweentherouteciphertranspositionandthesubstitutioncipherwherewholewordsaresubstitutedforlettersoftheoriginaltextmustbemadeonthebasisofthewordsactuallyuseditisbettertoconsidersuchamessageasaroutecipherwhenthewordsusedappeartohavesomeconsecutivemeaningbearingonthesituationathandasubstitutioncipherofthisvarietywouldonlybeusedfortransmissionofashortmessageofgreatimportanceandsecrecy"

#define KEY_LENGTH 30
#define SECTION_CONSTANT ENCRYPTEDLEN/KEY_LENGTH

#define HEUR_THRESHOLD_OP1 50
#define HEUR_THRESHOLD_OP2 70

#define OP1_HOP 4
#define OP2_HOP 2

#include <time.h>

bool extractBigrams(float *scores, const char* filename) {
  FILE* bigramsFile = fopen(filename, "r");
  if (bigramsFile == NULL) {
    fprintf(stderr, "Failed to open file %s. Exit\n", filename);
    return true;
  }
  while(1){
    char tempBigram[2];
    float tempBigramScore = 0.0;
    if (fscanf(bigramsFile, "%s %f", tempBigram, &tempBigramScore) < 2)
    { break; } 
    scores[(tempBigram[0]-'a')*ALPHABET + tempBigram[1]-'a'] = tempBigramScore; 
  }
  fclose(bigramsFile);
  return false;
}

bool verify(int* encrMap) {
  bool pass = true;
  const char *expect = DECRYPTED_T;
  for (int j=0; j<ENCRYPTEDLEN; ++j) {
    if (encrMap[j] + 'a' != expect[j]) {
       pass = false; break;
    }
  }
  return pass;
}

float candidateScore(int* decrMsg, float* scores) {
  float total = 0.0;
  for (int j=0; j<ENCRYPTEDLEN-1; ++j) 
    total += scores[ALPHABET*decrMsg[j] + decrMsg[j+1]];  
  return total;
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <path to file>\n", argv[0]);
    return 1;
  }
  const char* filename = argv[1];

  int encryptedMap[ENCRYPTEDLEN];

  for (int j=0; j<ENCRYPTEDLEN; ++j)
    encryptedMap[j] = ENCRYPTED_T[j] - 'a';

  float scores[totalBigrams];  
  bool fail = extractBigrams(scores, filename);
  if (fail) return 1;

  int* decrypted = new int[ENCRYPTEDLEN*THREADS];

  srand(time(NULL));

  omp_set_num_threads(B*T);

  float *scores_omp;
  int *encrypted_omp, *decrypted_omp;
  scores_omp = (float*)malloc(sizeof(float)*totalBigrams);
  encrypted_omp = (int*)malloc(sizeof(int)*ENCRYPTEDLEN);
  decrypted_omp = (int*)malloc(sizeof(int)*ENCRYPTEDLEN*THREADS);

  memcpy(scores_omp, scores, sizeof(float)*totalBigrams);
  memcpy(encrypted_omp, encryptedMap, sizeof(int)*ENCRYPTEDLEN);

  unsigned int* states;
  states = (unsigned int*)malloc(THREADS*sizeof(unsigned int));

  for (int i = 0; i < THREADS; i++)
    states[i] = i;

  double start = omp_get_wtime();

  #pragma omp parallel for
  for (int i = 0; i < B; i++) {
    #pragma omp parallel for
    for (int j = 0; j < T; j++) {
      int idx = i*T + j;
      unsigned int localState = states[idx];

      int key[KEY_LENGTH];
      for (int k = 0; k < KEY_LENGTH; k++)
        key[k] = k;

      for (int k = 0; k < KEY_LENGTH; k++) {
        int left = rand() % KEY_LENGTH;
        int right = rand() % KEY_LENGTH;
        key[left] = key[right];
        key[right] = key[left];
      }

      for (int cycles = 0; cycles < CLIMBINGS; cycles++) {
        int backupKey[KEY_LENGTH];
        for (int k = 0; k < KEY_LENGTH; k++)
          backupKey[k] = key[k];

        float tempScore = 0.0;

        int branch = rand() % 100;

        if (branch < HEUR_THRESHOLD_OP1) {
          for (int k = 0; k < 1 + rand() % OP1_HOP; k++) {
            int left = rand() % KEY_LENGTH;
            int right = rand() % KEY_LENGTH;
            key[left] = key[right];
            key[right] = key[left];
          }
        }

        else if (branch < HEUR_THRESHOLD_OP2) {
          for (int k = 0; k < 1 + rand() % OP2_HOP; k++) {
            int blockStart = rand() % KEY_LENGTH;
            int blockEnd = rand() % KEY_LENGTH;
            for (int l = 0; l < 1 + rand() % (abs(blockStart - blockEnd)); l++) {
              int left = (blockStart + l) % KEY_LENGTH;
              int right = (blockEnd + l) % KEY_LENGTH;
              key[left] = key[right];
              key[right] = key[left];
            }
          }
        }

        else {
          int l = 1 + rand() % (KEY_LENGTH - 2);
          int f = rand() % (KEY_LENGTH - 1);
          int t = (f + 1 + rand() % (KEY_LENGTH - 2)) % KEY_LENGTH;
          int t0 = (t - f + KEY_LENGTH) % KEY_LENGTH;
          int n = (t0 + l) % KEY_LENGTH;

          for (int k = 0; k < n; k++) {
            int ff = (f + k) % KEY_LENGTH;
            int tt = ((t0 + k) % n + f) % KEY_LENGTH;
            key[tt] = key[ff];
          }
        }

        int localDecrypted[ENCRYPTEDLEN];
        for (int k = 0; k < ENCRYPTEDLEN; k++)
          localDecrypted[k] = 0;

        for (int k = 0; k < KEY_LENGTH; k++) {
          int offset = 0;
          for (int l = 0; l < KEY_LENGTH; l++)
            if (key[l] < key[k])
              offset += ENCRYPTEDLEN / KEY_LENGTH;
          for (int l = 0; l < ENCRYPTEDLEN / KEY_LENGTH; l++) {
            localDecrypted[offset + l] = encrypted_omp[k * ENCRYPTEDLEN / KEY_LENGTH + l];
          }
        }

        for (int k = 0; k < ENCRYPTEDLEN - 1; k++) {
          tempScore += scores_omp[ALPHABET * localDecrypted[k] + localDecrypted[k + 1]];
        }

        if (tempScore < 999999.0) {
          for (int k = 0; k < ENCRYPTEDLEN; k++) {
            decrypted_omp[idx * ENCRYPTEDLEN + k] = localDecrypted[k];
          }
        }

        else {
          for (int k = 0; k < KEY_LENGTH; k++)
            key[k] = backupKey[k];
        }
      }
    }
  }

  double end = omp_get_wtime();
  printf("Kernel execution time %f (s)\n", end - start);

  int bestCandidate = 0;
  float bestScore = 999999.0;
  float* scoreHistory = (float*)malloc(B*T*sizeof(float));

  for (int i = 0; i < THREADS; i++) {
    float currentScore = candidateScore(&decrypted_omp[i * ENCRYPTEDLEN], scores_omp);
    scoreHistory[i] = currentScore;
    if (currentScore < bestScore) {
      bestScore = currentScore;
      bestCandidate = i;
    }
  }

  bool pass = verify(&decrypted_omp[bestCandidate * ENCRYPTEDLEN]);
  printf("%s\n", pass ? "PASS" : "FAIL");

  free(scores_omp);
  free(encrypted_omp);
  free(decrypted_omp);
  free(states);
  free(scoreHistory);
  delete[] decrypted;
  return 0;
}
```

Note that the code has been modified to use OpenMP for parallelization, and the CUDA-specific constructs have been replaced with OpenMP equivalents. The code also uses dynamic memory allocation to allocate memory for the OpenMP arrays. The `omp_set_num_threads` function is used to set the number of threads for the OpenMP parallel region. The `omp_get_wtime` function is used to measure the execution time of the OpenMP parallel region.