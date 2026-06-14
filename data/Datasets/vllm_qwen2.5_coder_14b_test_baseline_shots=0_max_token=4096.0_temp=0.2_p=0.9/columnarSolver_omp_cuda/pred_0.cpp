Below is the translated CUDA code from the provided OpenMP code. The main function remains unchanged, and all function names match the original.

```cpp
// main.cpp

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

__device__ float LCG_random_float(unsigned int * seed) {
  const unsigned int m = 2147483648;
  const unsigned int a = 26757677;
  const unsigned int c = 1;
  *seed = (a * (*seed) + c) % m;
  return (float) (*seed) / (float) m;
}

__device__ void LCG_random_init(unsigned int * seed) {
  const unsigned int m = 2147483648;
  const unsigned int a = 26757677;
  const unsigned int c = 1;
  *seed = (a * (*seed) + c) % m;
}

__device__ void decrypt(const int* encrypted, const int* key, int* decrypted) {

  int columns[KEY_LENGTH][SECTION_CONSTANT+1];
  int offset = 0;
  int colLength[KEY_LENGTH];

  for (int j=0; j<KEY_LENGTH; ++j) {
    colLength[j] = ENCRYPTEDLEN / KEY_LENGTH;
    if (j < ENCRYPTEDLEN % KEY_LENGTH)
      colLength[j]++;
  }

  for (int keyPos=0; keyPos < KEY_LENGTH; ++keyPos) {
    offset = 0;
    for (int i=0; i<KEY_LENGTH; ++i)
      if (key[i] < key[keyPos])
        offset += colLength[i];

    for (int j=0; j<colLength[keyPos]; ++j)   
      columns[key[keyPos]][j] = encrypted[offset+j];          
  } 

  for (int j=0; j<ENCRYPTEDLEN; ++j) 
    decrypted[j] = columns[key[j % KEY_LENGTH]][j / KEY_LENGTH];  
} 

__device__ void swapElements(int *key, int posLeft, int posRight) {
  if (posLeft != posRight)
  {
    key[posLeft] -= key[posRight];
    key[posRight] += key[posLeft];
    key[posLeft] = key[posRight] - key[posLeft];
  }
}

__device__ void swapBlock(int *key, int posLeft, int posRight, int length) {  
  for (int i=0; i<length; i++) 
    swapElements(key, (posLeft+i)%KEY_LENGTH, (posRight+i)%KEY_LENGTH);
}

__global__ void decodeKernel(
  const float *__restrict d_scores, 
  const int *__restrict d_encrypted,
  const unsigned int*__restrict globalState, 
  int *__restrict d_decrypted,
  float *__restrict shared_scores) {

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

  int lid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + lid;
  unsigned int localState = globalState[idx];

  if (lid == 0) {
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
      t = (f+1+(LCG_random_float(&localState)*(KEY_LENGTH-2)));
      t = t % KEY_LENGTH;

      for (j=0; j< KEY_LENGTH;j++)
        shiftHelper[j] = key[j];

      t0 = (t-f+KEY_LENGTH) % KEY_LENGTH;
      n = (t0+l) % KEY_LENGTH;

      for (j=0; j<n;j++) 
      {
        ff = (f+j) % KEY_LENGTH;
        tt = (((t0+j)%n)+f)%KEY_LENGTH;
        key[tt] = shiftHelper[ff];
      }        
    }      

    decrypt(d_encrypted, key, localDecrypted);    

    for (j=0; j<ENCRYPTEDLEN-1; ++j) {
      tempScore += shared_scores[ALPHABET*localDecrypted[j] + localDecrypted[j+1]];
    }

    if (tempScore < bestScore) {
      bestScore = tempScore;
      for (j=0; j<ENCRYPTEDLEN; ++j) {
        bestLocalDecrypted[j] = localDecrypted[j];
      }
    }    

    else 
    {
      for (j=0; j<KEY_LENGTH;j++)
        key[j] = backupKey[j];      
    }
  }

  for (j=0; j<ENCRYPTEDLEN; ++j)
    d_decrypted[idx*ENCRYPTEDLEN+j] = bestLocalDecrypted[j];
}

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

  float* d_scores;
  int* d_encrypted;
  int* d_decrypted;
  unsigned int* d_state;
  float* d_shared_scores;

  cudaMalloc(&d_scores, totalBigrams * sizeof(float));
  cudaMalloc(&d_encrypted, ENCRYPTEDLEN * sizeof(int));
  cudaMalloc(&d_decrypted, ENCRYPTEDLEN * THREADS * sizeof(int));
  cudaMalloc(&d_state, THREADS * sizeof(unsigned int));
  cudaMalloc(&d_shared_scores, ALPHABET * ALPHABET * sizeof(float));

  cudaMemcpy(d_scores, scores, totalBigrams * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_encrypted, encryptedMap, ENCRYPTEDLEN * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_state, state, THREADS * sizeof(unsigned int), cudaMemcpyHostToDevice);

  auto start = std::chrono::steady_clock::now();

  decodeKernel<<<B, T>>>(d_scores, d_encrypted, d_state, d_decrypted, d_shared_scores);

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Kernel execution time %f (s)\n", time * 1e-9f);

  cudaMemcpy(decrypted, d_decrypted, ENCRYPTEDLEN * THREADS * sizeof(int), cudaMemcpyDeviceToHost);

  cudaFree(d_scores);
  cudaFree(d_encrypted);
  cudaFree(d_decrypted);
  cudaFree(d_state);
  cudaFree(d_shared_scores);

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
  return 0;
}
```

This CUDA code replicates the functionality of the original OpenMP code, with the necessary changes to adapt it to CUDA's parallel programming model. The `decodeKernel` function is now a CUDA kernel, and memory management is handled using CUDA's memory allocation and copy functions.