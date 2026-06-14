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

  int* decrypted = new int[ENCRYPTEDLEN*THREADS];
  unsigned int state[THREADS];

  cudaMalloc((void**)&decrypted, ENCRYPTEDLEN*THREADS*sizeof(int));
  cudaMalloc((void**)&state, THREADS*sizeof(unsigned int));
  cudaMemset(state, 0, THREADS*sizeof(unsigned int));

  cudaMalloc((void**)&scores, totalBigrams*sizeof(float));
  cudaMemcpy(scores, scores, totalBigrams*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(encryptedMap, encryptedMap, ENCRYPTEDLEN*sizeof(int), cudaMemcpyHostToDevice);

  auto start = std::chrono::steady_clock::now();

  dim3 dimBlock(T);
  dim3 dimGrid(B);
  kernel<<<dimGrid, dimBlock>>>(scores, encryptedMap, state, decrypted);

  cudaDeviceSynchronize();

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Kernel execution time %f (s)\n", time * 1e-9f);

  int bestCandidate = 0;
  float bestScore = CAP;
  float* scoreHistory = new float[B*T];

  cudaMemcpy(scoreHistory, decrypted, B*T*sizeof(float), cudaMemcpyDeviceToHost);

  for (int j=0; j<THREADS; ++j)  {
    float currentScore = candidateScore(&decrypted[ENCRYPTEDLEN*j], scores);
    if (currentScore < bestScore) {
      bestScore = currentScore;
      bestCandidate = j;
    }
  }

  bool pass = verify(&decrypted[ENCRYPTEDLEN*bestCandidate]);
  printf("%s\n", pass ? "PASS" : "FAIL");

  delete[] decrypted;
  delete[] scoreHistory;
  cudaFree(decrypted);
  cudaFree(state);
  cudaFree(scores);
  return 0;
}

__global__ void kernel(float* scores, const int* encryptedMap, unsigned int* state, int* decrypted) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int localState = state[idx];

  if (idx < totalBigrams) {
    scores[idx] = 0.0f;
  }

  if (idx < ENCRYPTEDLEN) {
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

    for (j=0; j<ALPHABET;++j)
      for (jj=0; jj<ALPHABET; ++jj)
        scores[j*ALPHABET + jj] = 0.0f;

    for (j=0; j<KEY_LENGTH; ++j) 
      key[j]=j;

    for (j=0; j<KEY_LENGTH; ++j) {
      swapElements(key, j, (float)lcg_random_float(&localState)*KEY_LENGTH);
    }

    for (int cycles=0; cycles<CLIMBINGS; ++cycles) {  

      for (j=0; j<KEY_LENGTH;j++)
        backupKey[j] = key[j];

      tempScore = 0.f;

      int branch = (float)lcg_random_float(&localState)*100; 

      if (branch < HEUR_THRESHOLD_OP1)
      {
        for (j=0; j<1+(float)lcg_random_float(&localState)*OP1_HOP; j++) 
        {
          leftLetter = (float)lcg_random_float(&localState)*KEY_LENGTH;   
          rightLetter = (float)lcg_random_float(&localState)*KEY_LENGTH; 
          swapElements(key, leftLetter, rightLetter);
        }            
      }

      else if (branch < HEUR_THRESHOLD_OP2)
      {
        for (j=0; j< 1+(float)lcg_random_float(&localState)*OP2_HOP;j++)
        {
          blockStart = (float)lcg_random_float(&localState)*KEY_LENGTH;
          blockEnd = (float)lcg_random_float(&localState)*KEY_LENGTH;
          swapBlock(key, blockStart, blockEnd, 1+(float)lcg_random_float(&localState)*(abs((blockStart-blockEnd))-1));
        }
      }

      else 
      {
        l = 1 + (float)lcg_random_float(&localState)*(KEY_LENGTH-2);
        f = (float)lcg_random_float(&localState)*(KEY_LENGTH-1);
        t = (f+1+(float)lcg_random_float(&localState)*(KEY_LENGTH-2));
        t = t % KEY_LENGTH;

        for (j=0; j< KEY_LENGTH;j++)
          shiftHelper[j] = key[j];

        t0 = (t-f+KEY_LENGTH) % KEY_LENGTH;
        n = (t0+l) % KEY_LENGTH;

        for (j=0; j<n;j++) 
        {
          ff = (f+j)