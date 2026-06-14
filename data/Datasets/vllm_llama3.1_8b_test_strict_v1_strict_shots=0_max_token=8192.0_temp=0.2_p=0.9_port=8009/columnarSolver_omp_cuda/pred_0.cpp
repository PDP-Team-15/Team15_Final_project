```cuda
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define LCG_M 2147483648
#define LCG_A 26757677
#define LCG_C 1

__device__ float LCG_random_float(unsigned int * seed) {
  *seed = (LCG_A * (*seed) + LCG_C) % LCG_M;
  return (float) (*seed) / (float) LCG_M;
}

__device__ void LCG_random_init(unsigned int * seed) {
  *seed = (LCG_A * (*seed) + LCG_C) % LCG_M;
}

__global__ void decryptKernel(const int* __restrict d_encrypted, const int* __restrict d_key, int* __restrict d_decrypted) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int key[KEY_LENGTH];
  int localDecrypted[ENCRYPTEDLEN];  
  int leftLetter = 0;
  int rightLetter = 0;
  int backupKey[KEY_LENGTH];
  int shiftHelper[KEY_LENGTH];
  int blockStart, blockEnd;
  int l,f,t,t0,n,ff,tt;
  float tempScore = 0.f;
  float bestScore = CAP;
  int j = 0, jj = 0;

  if (idx < ENCRYPTEDLEN) {
    key[idx % KEY_LENGTH] = idx % KEY_LENGTH;
    for (j=0; j<KEY_LENGTH; ++j) {
      swapElements(key, j, LCG_random_float(&idx)*KEY_LENGTH);
    }

    for (int cycles=0; cycles<CLIMBINGS; ++cycles) {  

      for (j=0; j<KEY_LENGTH;j++)
        backupKey[j] = key[j];

      tempScore = 0.f;

      int branch = LCG_random_float(&idx)*100; 

      if (branch < HEUR_THRESHOLD_OP1)
      {
        for (j=0; j<1+LCG_random_float(&idx)*OP1_HOP; j++) 
        {
          leftLetter = LCG_random_float(&idx)*KEY_LENGTH;   
          rightLetter = LCG_random_float(&idx)*KEY_LENGTH; 
          swapElements(key, leftLetter, rightLetter);
        }            
      }

      else if (branch < HEUR_THRESHOLD_OP2)
      {
        for (j=0; j< 1+LCG_random_float(&idx)*OP2_HOP;j++)
        {
          blockStart = LCG_random_float(&idx)*KEY_LENGTH;
          blockEnd = LCG_random_float(&idx)*KEY_LENGTH;
          swapBlock(key, blockStart, blockEnd, 1+LCG_random_float(&idx)*(abs((blockStart-blockEnd))-1));
        }
      }

      else 
      {
        l = 1 + LCG_random_float(&idx)*(KEY_LENGTH-2);
        f = LCG_random_float(&idx)*(KEY_LENGTH-1);
        t = (f+1+(LCG_random_float(&idx)*(KEY_LENGTH-2)));
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
        tempScore += d_decrypted[idx*ENCRYPTEDLEN+j*ALPHABET + (idx*ENCRYPTEDLEN+j+1)];
      }

      if (tempScore < bestScore) {
        bestScore = tempScore;
        for (j=0; j<ENCRYPTEDLEN; ++j) {
          d_decrypted[idx*ENCRYPTEDLEN+j] = localDecrypted[j];
        }
      }    

      else 
      {
        for (j=0; j<KEY_LENGTH;j++)
          key[j] = backupKey[j];      
      }
    }
  }
}

__global__ void decodeKernel(const float* __restrict d_scores, const int* __restrict d_encrypted, const unsigned int*__restrict d_globalState, int* __restrict d_decrypted, float* __restrict shared_scores) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
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

  if (idx < ENCRYPTEDLEN) {
    key[idx % KEY_LENGTH] = idx % KEY_LENGTH;
    for (j=0; j<KEY_LENGTH; ++j) {
      swapElements(key, j, LCG_random_float(&idx)*KEY_LENGTH);
    }

    for (int cycles=0; cycles<CLIMBINGS; ++cycles) {  

      for (j=0; j<KEY_LENGTH;j++)
        backupKey[j] = key[j];

      tempScore = 0.f;

      int branch = LCG_random_float(&idx)*100; 

      if (branch < HEUR_THRESHOLD_OP1)
      {
        for (j=0; j<1+LCG_random_float(&idx)*OP1_HOP; j++) 
        {
          leftLetter = LCG_random_float(&idx)*KEY_LENGTH;   
          rightLetter = LCG_random_float(&idx)*KEY_LENGTH; 
          swapElements(key, leftLetter, rightLetter);
        }            
      }

      else if (branch < HEUR_THRESHOLD_OP2)
      {
        for (j=0; j< 1+LCG_random_float(&idx)*OP2_HOP;j++)
        {
          blockStart = LCG_random_float(&idx)*KEY_LENGTH;
          blockEnd = LCG_random_float(&idx)*KEY_LENGTH;
          swapBlock(key, blockStart, blockEnd, 1+LCG_random_float(&idx)*(abs((blockStart-blockEnd))-1));
        }
      }

      else 
      {
        l = 1 + LCG_random_float(&idx)*(KEY_LENGTH-2);
        f = LCG_random_float(&idx)*(KEY_LENGTH-1);
        t = (f+1+(LCG_random_float(&idx)*(KEY_LENGTH-2)));
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
        tempScore += d_scores[j*ALPHABET + localDecrypted[j+1]];
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

    if (threadIdx.x == 0) {
      for (j=0; j<ALPHABET;++j)
        for (jj=0; jj<ALPHABET; ++jj)
          shared_scores[j*ALPHABET + jj] = d_scores[j*ALPHABET + jj];
    }

    __syncthreads();

    for (j=0; j<ENCRYPTEDLEN; ++j)
      d_decrypted[idx*ENCRYPTEDLEN+j] = bestLocalDecrypted[j];
  }
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

__global__ void extractBigramsKernel(float* __restrict d_scores, const char* __restrict filename) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  FILE* bigramsFile = fopen(filename, "r");
  if (bigramsFile == NULL) {
    printf("Failed to open file %s. Exit\n", filename);
    return;
  }
  while(1){
    char tempBigram[2];
    float tempBigramScore = 0.0;
    if (fscanf(bigramsFile, "%s %f", tempBigram, &tempBigramScore) < 2)
    { break; } 
    d_scores[(tempBigram[0]-'a')*ALPHABET + tempBigram[1]-'a'] = tempBigramScore; 
  }
  fclose(bigramsFile);
}

__global__ void verifyKernel(int* __restrict d_encrMap) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const char *expect = DECRYPTED_T;
  bool pass = true;
  for (int j=0; j<ENCRYPTEDLEN; ++j) {
    if (d_encrMap[j] + 'a' != expect[j]) {
       pass = false; break;
    }
  }
  if (idx == 0) {
    printf("%s\n", pass ? "PASS" : "FAIL");
  }
}

__global__ void candidateScoreKernel(float* __restrict d_decrypted, float* __restrict d_scores) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  float total = 0.0;
  for (int j=0; j<ENCRYPTEDLEN-1; ++j) 
    total += d_scores[ALPHABET*d_decrypted[idx*ENCRYPTEDLEN+j] + d_decrypted[idx*ENCRYPTEDLEN+j+1]];  
  if (idx == 0) {
    printf("Kernel execution time %f (s)\n", 0.0f);
  }
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
  bool fail = extractBigramsKernel<<<1, 1>>>(scores, filename);
  cudaDeviceSynchronize();
  if (fail) return 1;

  int* decrypted = (int*)malloc(ENCRYPTEDLEN*THREADS*sizeof(int));
  unsigned int state[THREADS];

  cudaMalloc((void**)&d_encrypted, ENCRYPTEDLEN*sizeof(int));
  cudaMalloc((void**)&d_key, KEY_LENGTH*sizeof(int));
  cudaMalloc((void**)&d_decrypted, ENCRYPTEDLEN*THREADS*sizeof(int));
  cudaMalloc((void**)&d_scores, totalBigrams*sizeof(float));
  cudaMalloc((void**)&d_globalState, THREADS*sizeof(unsigned int));
  cudaMalloc((void**)&d_shared_scores, ALPHABET*ALPHABET*sizeof(float));

  cudaMemcpy(d_encrypted, ENCRYPTED_T, ENCRYPTEDLEN*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_key, ENCRYPTED_T, KEY_LENGTH*sizeof(int), cudaMemcpyHostToDevice);

  for (int i=0; i<THREADS; ++i) {
    state[i] = i;
    LCG_random_init(&state[i]);
  }
  cudaMemcpy(d_globalState, state, THREADS*sizeof(unsigned int), cudaMemcpyHostToDevice);

  for (int i=0; i<THREADS; ++i) {
    decrypted[i*ENCRYPTEDLEN] = ENCRYPTED_T[i % KEY_LENGTH] - 'a';
  }
  cudaMemcpy(d_decrypted, decrypted, ENCRYPTEDLEN*THREADS*sizeof(int), cudaMemcpyHostToDevice);

  cudaMemcpy(d_scores, scores, totalBigrams*sizeof(float), cudaMemcpyHostToDevice);

  int threads = 256;
  int blocks = (THREADS + threads - 1) / threads;
  decodeKernel<<<blocks, threads>>>(d_scores, d_encrypted, d_globalState, d_decrypted, d_shared_scores);
  cudaDeviceSynchronize();

  float bestScore = CAP;
  float* scoreHistory = (float*)malloc(B*T*sizeof(float));

  for (int i=0; i<THREADS; ++i)  {
    float currentScore = candidateScoreKernel<<<1, 1>>>(d_decrypted + i*ENCRYPTEDLEN, d_scores);
    cudaDeviceSynchronize();
    cudaMemcpy(&scoreHistory[i], &currentScore, sizeof(float), cudaMemcpyDeviceToHost);
    if (scoreHistory[i] < bestScore) {
      bestScore = scoreHistory[i];
    }
  }

  int bestCandidate = 0;
  for (int i=0; i<THREADS; ++i) {
    if (scoreHistory[i] == bestScore) {
      bestCandidate = i;
      break;
    }
  }

  verifyKernel<<<1, 1>>>(d_encrypted + bestCandidate*ENCRYPTEDLEN);
  cudaDeviceSynchronize();

  free(decrypted);
  free(scoreHistory);
  return 0;
}
```
Note that the CUDA code uses the `cudaMalloc` and `cudaMemcpy` functions to manage memory on the device, and the `cudaDeviceSynchronize` function to ensure that all kernel launches have completed before accessing the results. The `cudaFree` function is not used in this example, as the memory is freed automatically when the program exits.