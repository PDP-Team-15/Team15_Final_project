Here is the translation of the CUDA code to OpenMP:

```cpp
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
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

#include "kernels.cpp"

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

  int *encrypted = new int[ENCRYPTEDLEN];
  memcpy(encrypted, encryptedMap, sizeof(int)*ENCRYPTEDLEN);

  int* decrypted = new int[ENCRYPTEDLEN*THREADS];

  #pragma omp target data map(to:scores[:totalBigrams], encrypted[:ENCRYPTEDLEN]) map(from:decrypted[:ENCRYPTEDLEN*THREADS])

  #pragma omp parallel num_threads(B*T)
  {
    unsigned int localState = omp_get_thread_num();

    #pragma omp single nowait
    {
      setupKernel(localState);
    }

    #pragma omp single nowait
    {
      decode(scores, encrypted, localState, decrypted);
    }
  }

  int bestCandidate = 0;
  float bestScore = CAP;
  float* scoreHistory = new float[B*T];

  for (int j=0; j<THREADS; ++j)  {
    float currentScore = candidateScore(&decrypted[ENCRYPTEDLEN*j], scores);
    scoreHistory[j] = currentScore;
    if (currentScore < bestScore) {
      bestScore = currentScore;
      bestCandidate = j;
    }    
  }  

  bool pass = verify(&decrypted[ENCRYPTEDLEN*bestCandidate]);
  printf("%s\n", pass ? "PASS" : "FAIL");

  delete[] encrypted;
  delete[] decrypted;
  delete[] scoreHistory;
  return 0;
}

// kernels.cpp remains unchanged as it does not contain any CUDA-specific code
```

### Explanation:
1. **Memory Management**: The memory allocation and deallocation remain the same since they are not specific to CUDA.
2. **Random Number Generation**: The `LCG_random_float` and `LCG_random_init` functions are kept as they are since they do not depend on CUDA's random number generation capabilities.
3. **Setup Kernel**: This function initializes the random states for each thread. In OpenMP, this is done using the `#pragma omp single` directive.
4. **Decode Kernel**: This function performs the decryption and scoring. It uses OpenMP's parallel regions to distribute the work across threads. The `#pragma omp parallel` directive specifies the number of threads, and the `#pragma omp single` directive ensures that certain operations (like setting up the initial key) are performed by only one thread.

This translation should provide the same functionality as the original CUDA code but using OpenMP instead.