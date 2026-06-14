Here's the equivalent OpenMP code for the provided CUDA code:

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

#include "kernels.h"

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

unsigned int LCG_random_float(unsigned int * seed) {
  const unsigned int m = 2147483648;
  const unsigned int a = 26757677;
  const unsigned int c = 1;
  *seed = (a * (*seed) + c) % m;
  return (unsigned int) (*seed) / (unsigned int) m;
}

void LCG_random_init(unsigned int * seed) {
  const unsigned int m = 2147483648;
  const unsigned int a = 26757677;
  const unsigned int c = 1;
  *seed = (a * (*seed) + c) % m;
}

void setupKernel(unsigned int* state, int num_threads) {
  #pragma omp parallel for
  for (int i = 0; i < num_threads; i++) {
    state[i] = i;
    for (int j = 0; j < i; j++)
      LCG_random_init(&state[i]);
  }
}

void decrypt(const int* encrypted, const int* key, int* decrypted, int num_threads) {
  int columns[KEY_LENGTH][SECTION_CONSTANT+1];
  int offset = 0;
  int colLength[KEY_LENGTH];

  for (int j=0; j<KEY_LENGTH; ++j) {
    colLength[j] = ENCRYPTEDLEN / KEY_LENGTH;
    if (j < ENCRYPTEDLEN % KEY_LENGTH)
      colLength[j]++;
  }

  #pragma omp parallel for
  for (int i = 0; i < num_threads; i++) {
    for (int keyPos=0; keyPos < KEY_LENGTH; ++keyPos) {
      offset = 0;
      for (int j=0; j<KEY_LENGTH; ++j)
        if (key[j] < key[keyPos])
          offset += colLength[j];

      for (int j=0; j<colLength[keyPos]; ++j)   
        columns[key[keyPos]][j] = encrypted[offset+j];          
    } 

    for (int j=0; j<ENCRYPTEDLEN; ++j) 
      decrypted[i*ENCRYPTEDLEN+j] = columns[key[j % KEY_LENGTH]][j / KEY_LENGTH];  
  } 
}

void swapElements(int *key, int posLeft, int posRight) {
  if (posLeft != posRight)
  {
    key[posLeft] -= key[posRight];
    key[posRight] += key[posLeft];
    key[posLeft] = key[posRight] - key[posLeft];
  }
}

void swapBlock(int *key, int posLeft, int posRight, int length) {  
  for (int i=0; i<length; i++) 
    swapElements(key, (posLeft+i)%KEY_LENGTH, (posRight+i)%KEY_LENGTH);
}

void decode(const float *__restrict d_scores, 
            const int *__restrict d_encrypted,
            const unsigned int*__restrict  globalState, 
            int *__restrict d_decrypted, int num_threads) {

  __shared__ float shared_scores[ALPHABET*ALPHABET];

  int key[KEY_LENGTH];
  int localDecrypted[ENCRYPTEDLEN];  
  int bestLocalDecrypted[ENCRYPTEDLEN];  
  int leftLetter = 0;
  int rightLetter = 0;
  int backupKey[KEY_LENGTH];
  int shiftHelper[KEY_LENGTH];
  int blockStart, blockEnd;
  int l,f,t,t0,n,ff,tt;
  float tempScore = 0.f;
  float bestScore = CAP;
  int j = 0, jj = 0;

  #pragma omp parallel for
  for (int i = 0; i < num_threads; i++) {
    int idx = i;
    unsigned int localState = globalState[idx];

    if (idx == 0) {
      for (j=0; j<ALPHABET;++j)
        for (jj=0; jj<ALPHABET; ++jj)
          shared_scores[j*ALPHABET + jj] = d_scores[j*ALPHABET + jj];
    }

    __syncthreads();

    for (j=0; j<KEY_LENGTH; ++j) 
      key[j]=j;

    for (j=0; j<KEY_LENGTH; ++j) {
      swapElements(key, j, LCG_random_float(&localState)*KEY_LENGTH);
    }

    for (int cycles=0; cycles<CLIMBINGS; ++cycles) {  

      for (j=0; j<KEY_LENGTH;j++)
        backupKey[j] = key[j];

      tempScore = 0.f;

      int branch = LCG_random_float(&localState)*100; 

      if (branch < HEUR_THRESHOLD_OP1)
      {
        for (j=0; j<1+LCG_random_float(&localState)*OP1_HOP; j++) 
        {
          leftLetter = LCG_random_float(&localState)*KEY_LENGTH;   
          rightLetter = LCG_random_float(&localState)*KEY_LENGTH; 
          swapElements(key, leftLetter, rightLetter);
        }            
      }

      else if (branch < HEUR_THRESHOLD_OP2)
      {
        for (j=0; j< 1+LCG_random_float(&localState)*OP2_HOP;j++)
        {
          blockStart = LCG_random_float(&localState)*KEY_LENGTH;
          blockEnd = LCG_random_float(&localState)*KEY_LENGTH;
          swapBlock(key, blockStart, blockEnd, 1+LCG_random_float(&localState)*(abs((blockStart-blockEnd))-1));
        }
      }

      else 
      {
        l = 1 + LCG_random_float(&localState)*(KEY_LENGTH-2);
        f = LCG_random_float(&localState)*(KEY_LENGTH-1);
        t = (f+1+(LCG_random_float(&localState)*(KEY_LENGTH-