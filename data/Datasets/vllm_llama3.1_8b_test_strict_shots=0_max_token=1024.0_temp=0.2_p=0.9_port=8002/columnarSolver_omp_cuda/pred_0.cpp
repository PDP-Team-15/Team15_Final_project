```cpp
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <chrono>
#include <cuda_runtime.h>

#define B ((int)32)
#define T ((int)32)
#define THREADS ((int)B*T)
#define CLIMBINGS 150000
#define ALPHABET 26
#define totalBigrams ((int)ALPHABET*ALPHABET)
#define CAP ((float)999999.0)

#define ENCRYPTED_T "tteohtedanisroudesereguwocubsoitoabbofeiaiutsdheeisatsarsturesuaastniersrotnesctrctxdiwmhcusyenorndasmhaipnnptmaeecspegdeislwoheoiymreeotbsspiatoanihrelhwctftrhpuunhoianunreetrioettatlsnehtbaecpvgtltcirottonesnobeeeireaymrtohaawnwtesssvassirsrhabapnsynntitsittchitoosbtelmlaouitrehhwfeiaandeitciegfreoridhdcsheucrnoihdeoswobaceeaorgndlstigeearsotoetduedininttpedststntefoeaheoesuetvmmiorftuuhsurof"
#define ENCRYPTEDLEN ((int)sizeof(ENCRYPTED_T)-1)

#define DECRYPTED_T "thedistinctionbetweentherouteciphertranspositionandthesubstitutioncipherwherewholewordsaresubstitutedforlettersoftheoriginaltextmustbemadeonthebasisofthewordsactuallyuseditisbettertoconsidersuchamessageasaroutecipherwhenthewordsusedappeartohavesomeconsecutivemeaningbearingonthesituationathendasubstitutioncipherofthisvarietywouldonlybeusedfortransmissionofashortmessageofgreatimportanceandsecrecy"

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

  int* decrypted = new int[ENCRYPTEDLEN*THREADS];
  unsigned int state[THREADS];

  cudaMalloc((void**)&decrypted, ENCRYPTEDLEN*THREADS*sizeof(int));
  cudaMemset(decrypted, 0, ENCRYPTEDLEN*THREADS*sizeof(int));

  cudaMalloc((void**)&state, THREADS*sizeof(unsigned int));
  cudaMemset(state, 0, THREADS*sizeof(unsigned int));

  cudaMalloc((void**)&scores, totalBigrams*sizeof(float));
  cudaMemcpy(scores, &scores[0], total