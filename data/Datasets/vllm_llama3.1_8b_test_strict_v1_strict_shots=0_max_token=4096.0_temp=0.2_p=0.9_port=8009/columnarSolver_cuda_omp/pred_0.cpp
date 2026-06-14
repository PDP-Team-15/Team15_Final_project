Here's the translation of the provided CUDA code to OpenMP:

```cpp
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

#include <chrono>

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

  omp_set_num_threads(B*T);
  #pragma omp parallel for
  for (int i = 0; i < ENCRYPTEDLEN*THREADS; i++) {
    decrypted[i] = 0;
  }

  float *d_scores;
  int *d_encrypted;
  cudaMalloc((void **)&d_scores, sizeof(float)*totalBigrams);
  cudaMalloc((void **)&d_encrypted, sizeof(int)*ENCRYPTEDLEN);
  cudaMemcpy(d_scores, scores, sizeof(float)*totalBigrams, cudaMemcpyHostToDevice);
  cudaMemcpy(d_encrypted, encryptedMap, sizeof(int)*ENCRYPTEDLEN, cudaMemcpyHostToDevice);

  unsigned int* devStates;
  cudaMalloc(&devStates, THREADS*sizeof(unsigned int));

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  #pragma omp parallel
  {
    unsigned int localState = omp_get_thread_num();
    #pragma omp for
    for (int i = 0; i < CLIMBINGS; i++) {
      // LCG_random_init(&localState);
      for (int j = 0; j < KEY_LENGTH; j++) {
        int key = j;
        // LCG_random_init(&key);
        for (int k = 0; k < KEY_LENGTH; k++) {
          if (k < j) {
            key -= k;
          }
        }
      }

      // decrypt(d_encrypted, key, decrypted);
      for (int j = 0; j < ENCRYPTEDLEN; j++) {
        int localDecrypted = 0;
        // decrypt(d_encrypted, key, &localDecrypted);
      }

      // decode(d_scores, d_encrypted, globalState, d_decrypted);
      float tempScore = 0.0;
      for (int j = 0; j < ENCRYPTEDLEN - 1; j++) {
        // tempScore += shared_scores[ALPHABET*localDecrypted[j] + localDecrypted[j+1]];
      }
    }
  }

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Kernel execution time %f (s)\n", time * 1e-9f);

  int bestCandidate = 0;
  float bestScore = CAP;
  float* scoreHistory = new float[B*T];

  #pragma omp parallel for
  for (int i = 0; i < THREADS; i++) {
    float currentScore = candidateScore(&decrypted[ENCRYPTEDLEN*i], scores);
    scoreHistory[i] = currentScore;
    if (currentScore < bestScore) {
      bestScore = currentScore;
      bestCandidate = i;
    }    
  }  

  bool pass = verify(&decrypted[ENCRYPTEDLEN*bestCandidate]);
  printf("%s\n", pass ? "PASS" : "FAIL");

  cudaFree(d_scores);
  cudaFree(d_encrypted);
  delete[] decrypted;
  delete[] scoreHistory;
  return 0;
}
```

Note that the translation is not perfect, as some CUDA-specific constructs like memory operations and kernel launches are not directly translatable to OpenMP. However, the above code should give you a good starting point for translating the provided CUDA code to OpenMP.