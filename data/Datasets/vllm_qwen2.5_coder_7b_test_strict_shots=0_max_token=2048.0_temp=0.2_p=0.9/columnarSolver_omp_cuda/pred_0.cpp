```cpp
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
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

const char* ENCRYPTED_T = "tteohtedanisroudesereguwocubsoitoabbofeiaiutsdheeisatsarsturesuaastniersrotnesctrctxdiwmhcusyenorndasmhaipnnptmaeecspegdeislwoheoiymreeotbsspiatoanihrelhwctftrhpuunhoianunreetrioettatlsnehtbaecpvgtltcirottonesnobeeeireaymrtohaawnwtesssvassirsrhabapnsynntitsittchitoosbtelmlaouitrehhwfeiaandeitciegfreoridhdcsheucrnoihdeoswobaceeaorgndlstigeearsotoetduedininttpedststntefoeaheoesuetvmmiorftuuhsurof";
const int ENCRYPTEDLEN = sizeof(ENCRYPTED_T)-1;

const char* DECRYPTED_T = "thedistinctionbetweentherouteciphertranspositionandthesubstitutioncipherwherewholewordsaresubstitutedforlettersoftheoriginaltextmustbemadeonthebasisofthewordsactuallyuseditisbettertoconsidersuchamessageasaroutecipherwhenthewordsusedappeartohavesomeconsecutivemeaningbearingonthesituationathandasubstitutioncipherofthisvarietywouldonlybeusedfortransmissionofashortmessageofgreatimportanceandsecrecy";

const int KEY_LENGTH = 30;
const int SECTION_CONSTANT = ENCRYPTEDLEN/KEY_LENGTH;

const int HEUR_THRESHOLD_OP1 = 50;
const int HEUR_THRESHOLD_OP2 = 70;

const int OP1_HOP = 4;
const int OP2_HOP = 2;

__global__ void decodeKernel(
  const float *d_scores, 
  const int *d_encrypted,
  const unsigned int *globalState, 
  int *d_decrypted,
  float *shared_scores) {

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

  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int idx = bid * blockDim.x + tid;
  unsigned int localState = globalState[idx];

  if (tid == 0) {
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
    if (j < ENCRYPTEDLEN