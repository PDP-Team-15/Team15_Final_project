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

#include "kernels.cu"

__global__ void extractBigramsKernel(float *scores, const char* filename) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < totalBigrams) {
    FILE* bigramsFile = fopen(filename, "r");
    if (bigramsFile == NULL) {
      fprintf(stderr, "Failed to open file %s. Exit\n", filename);
      return;
    }
    while(1){
      char tempBigram[2];
      float tempBigramScore = 0.0;
      if (fscanf(bigramsFile, "%s %f", tempBigram, &tempBigramScore) < 2)
      { break; } 
      scores[idx] = tempBigramScore; 
    }
    fclose(bigramsFile);
  }
}

__global__ void verifyKernel(int* encrMap) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < ENCRYPTEDLEN) {
    const char *expect = DECRYPTED_T;
    bool pass = true;
    for (int j=0; j<ENCRYPTEDLEN; ++j) {
      if (encrMap[j] + 'a' != expect[j]) {
         pass = false; break;
      }
    }
    if (pass) {
      printf("PASS\n");
    } else {
      printf("FAIL\n");
    }
  }
}

__global__ void candidateScoreKernel(int* decrMsg, float* scores) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < THREADS) {
    float total = 0.0;
    for (int j=0; j<ENCRYPTEDLEN-1; ++j) 
      total += scores[ALPHABET*decrMsg[idx*ENCRYPTEDLEN+j] + decrMsg[idx*ENCRYPTEDLEN+j+1]];  
    printf("Thread %d: %f\n", idx, total);
  }
}

__global__ void decryptKernel(int* encrypted, int* key, int* decrypted) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < THREADS) {
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
      decrypted[idx*ENCRYPTEDLEN+j] = columns[key[j % KEY_LENGTH]][j / KEY_LENGTH];  
  }
}

__global__ void decodeKernel(
  const float *__restrict d_scores, 
    const int *__restrict d_encrypted,
  const unsigned int*__restrict globalState, 
          int *__restrict d_decrypted,
        float *__restrict shared_scores) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int localState = globalState[idx];

  if (threadIdx.x == 0) {
    for (int j=0; j<ALPHABET;++j)
      for (int jj=0; jj<ALPHABET; ++jj)
        shared_scores[j*ALPHABET + jj] = d_scores[j*ALPHABET + jj];
  }

  __syncthreads();

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

  for (j=0; j<KEY_LENGTH; ++j) 
    key[j]=j;

  for (j=0; j<KEY_LENGTH; ++j) {
    int rand = (int)(localState * 1000000) % KEY_LENGTH;
    swapElements(key, j, rand);
  }

  for (int cycles=0; cycles<CLIMBINGS; ++cycles) {  

    for (j=0; j<KEY_LENGTH;j++)
      backupKey[j] = key[j];

    tempScore = 0.f;

    int branch = (int)(localState * 100) % 100; 

    if (branch < HEUR_THRESHOLD_OP1)
    {
      for (j=0; j<1+(int)(localState * OP1_HOP); j++) 
      {
        leftLetter = (int)(localState * KEY_LENGTH) % KEY_LENGTH;   
        rightLetter = (int)(localState * KEY_LENGTH) % KEY_LENGTH; 
        swapElements(key, leftLetter, rightLetter);
      }            
    }

    else if (branch < HEUR_THRESHOLD_OP2)
    {
      for (j=0; j< 1+(int)(localState * OP2_HOP);j++)
      {
        blockStart = (int)(localState * KEY_LENGTH) % KEY_LENGTH;
        blockEnd = (int)(localState * KEY_LENGTH) % KEY_LENGTH;
        swapBlock(key, blockStart, blockEnd, 1+(int)(localState * (abs((blockStart-blockEnd))-1)));
      }
    }

    else 
    {
      l = 1 + (int)(localState * (KEY_LENGTH-2));
      f = (int)(localState * (KEY_LENGTH-1));
      t = (f+1+(int)(localState * (KEY_LENGTH-2)));
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

    decryptKernel<<<1, THREADS>>>(d_encrypted, key, localDecrypted);    

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

  d_decrypted[idx*ENCRYPTEDLEN] = bestLocalDecrypted[0];
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
  bool fail = extractBigramsKernel<<<1, 256>>>(scores, filename);
  cudaDeviceSynchronize();
  if (fail) return 1;

  int* decrypted = new int[ENCRYPTEDLEN*THREADS];
  unsigned int* state = new unsigned int[THREADS];

  cudaMalloc((void**)&decrypted, ENCRYPTEDLEN*THREADS*sizeof(int));
  cudaMalloc((void**)&state, THREADS*sizeof(unsigned int));

  cudaMemcpy(state, state, THREADS*sizeof(unsigned int), cudaMemcpyHostToDevice);

  decodeKernel<<<B, T>>>(scores, encryptedMap, state, decrypted, NULL);

  cudaDeviceSynchronize();

  int bestCandidate = 0;
  float bestScore = CAP;
  float* scoreHistory = new float[B*T];

  cudaMemcpy(scoreHistory, scoreHistory, B*T*sizeof(float), cudaMemcpyDeviceToHost);

  for (int j=0; j<THREADS; ++j)  {
    float currentScore = candidateScoreKernel<<<1, 256>>>(decrypted, scores);
    cudaDeviceSynchronize();
    printf("Thread %d: %f\n", j, currentScore);
    if (currentScore < bestScore) {
      bestScore = currentScore;
      bestCandidate = j;
    }
  }

  verifyKernel<<<1, 256>>>(decrypted);

  cudaDeviceSynchronize();

  bool pass = true;
  cudaMemcpy(&pass, &pass, sizeof(bool), cudaMemcpyDeviceToHost);
  printf("%s\n", pass ? "PASS" : "FAIL");

  delete[] decrypted;
  delete[] scoreHistory;
  return 0;
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

float LCG_random_float(unsigned int * seed) {
  const unsigned int m = 2147483648;
  const unsigned int a = 26757677;
  const unsigned int c = 1;
  *seed = (a * (*seed) + c) % m;
  return (float) (*seed) / (float) m;
}

void LCG_random_init(unsigned int * seed) {
  const unsigned int m = 2147483648;
  const unsigned int a = 26757677;
  const unsigned int c = 1;
  *seed = (a * (*seed) + c) % m;
}

void decrypt(const int* encrypted, const int* key, int* decrypted) {

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

void kernels_init() {
  cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
  cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeDefault);
  cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1024*1024*1024);
  cudaDeviceSetLimit(cudaLimitDevRuntimeMemory, 1024*1024*1024);
}

void kernels_cleanup() {
  cudaDeviceReset();
}
```

```cpp
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

__global__ void kernels_init() {
  cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
  cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeDefault);
  cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1024*1024*1024);
  cudaDeviceSetLimit(cudaLimitDevRuntimeMemory, 1024*1024*1024);
}

__global__ void kernels_cleanup() {
  cudaDeviceReset();
}
```

```cpp
int main(int argc, char* argv[]) {
  kernels_init();
  // your code here
  kernels_cleanup();
  return 0;
}
```