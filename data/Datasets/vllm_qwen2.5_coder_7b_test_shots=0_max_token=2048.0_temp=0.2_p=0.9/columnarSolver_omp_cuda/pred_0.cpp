Here is the CUDA translation of the provided OpenMP code:

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

#include "kernels.cu"

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

  cudaMallocManaged(&decrypted, ENCRYPTEDLEN*THREADS*sizeof(int));
  cudaMallocManaged(&state, THREADS*sizeof(unsigned int));

  auto start = std::chrono::steady_clock::now();

  dim3 blocks(B);
  dim3 threads(T);

  decodeKernel<<<blocks, threads>>>(scores, encryptedMap, state, decrypted);

  cudaDeviceSynchronize();

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Kernel execution time %f (s)\n", time * 1e-9f);

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

  delete[] decrypted;
  delete[] scoreHistory;
  cudaFree(decrypted);
  cudaFree(state);
  return 0;
}
```

And here is the corresponding `kernels.cu` file:

```cpp
#include <curand_kernel.h>

__global__ void LCG_random_init(unsigned int * seed) {
  const unsigned int m = 2147483648;
  const unsigned int a = 26757677;
  const unsigned int c = 1;
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  seed[tid] = (a * seed[tid] + c) % m;
}

__global__ void decodeKernel(
  const float *__restrict d_scores, 
  const int *__restrict d_encrypted,
  const unsigned int*__restrict globalState, 
  int *__restrict d_decrypted,
  float *__restrict shared_scores) {

  extern __shared__ float s_scores[];

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

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int localState = globalState[tid];

  if (tid == 0) {
    for (j=0; j<ALPHABET;++j)
      for (jj=0; jj<ALPHABET; ++jj)
        s_scores[j*ALPHABET + jj] = d_scores[j*ALPHABET + jj];
  }

  __syncthreads();

  for (j=0; j<KEY_LENGTH; ++j) 
    key[j]=j;

  for (j=0; j<KEY_LENGTH; ++j) {
    swapElements(key, j, curand_uniform(&localState)*KEY_LENGTH);
  }

  for (int cycles=0; cycles<CLIMBINGS; ++cycles) {  

    for (j=0; j<KEY_LENGTH;j++)
      backupKey[j] = key[j];

    tempScore = 0.f;

    int branch = curand_uniform(&localState)*100; 

    if (branch < HEUR_THRESHOLD_OP1)
    {
      for (j=0; j<1+curand_uniform(&localState)*OP1_HOP; j++) 
      {
        leftLetter = curand_uniform(&localState)*KEY_LENGTH;   
        rightLetter = curand_uniform(&localState)*KEY_LENGTH; 
        swapElements(key, leftLetter, rightLetter);
      }            
    }

    else if (branch < HEUR_THRESHOLD_OP2)
    {
      for (j=0; j< 1+curand_uniform(&localState)*OP2_HOP;j++)
      {
        blockStart = curand_uniform(&localState)*KEY_LENGTH;
        blockEnd = curand_uniform(&localState)*KEY_LENGTH;
        swapBlock(key, blockStart, blockEnd, 1+curand_uniform(&localState)*(abs((blockStart-blockEnd))-1));
      }
    }

    else 
    {
      l = 1 + curand_uniform(&localState)*(KEY_LENGTH-2);
      f = curand_uniform(&localState)*(KEY_LENGTH-1);
      t = (f+1+(curand_uniform(&localState)*(KEY_LENGTH-2)));
      t = t % KEY_LENGTH;

      for (j=0; j< KEY_LENGTH;j++)
        shiftHelper[j] = key[j];

      t0 = (t