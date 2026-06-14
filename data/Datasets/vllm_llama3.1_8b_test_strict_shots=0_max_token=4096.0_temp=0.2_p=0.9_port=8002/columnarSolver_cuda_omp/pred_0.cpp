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

  #pragma omp parallel for num_threads(THREADS)
  for (int j=0; j<ENCRYPTEDLEN*THREADS; ++j)
    decrypted[j] = -1;

  float *scores_omp;
  int *encryptedMap_omp;
  int *decrypted_omp;

  scores_omp = (float*)malloc(sizeof(float)*totalBigrams);
  encryptedMap_omp = (int*)malloc(sizeof(int)*ENCRYPTEDLEN);
  decrypted_omp = (int*)malloc(sizeof(int)*ENCRYPTEDLEN*THREADS);

  memcpy(scores_omp, scores, sizeof(float)*totalBigrams);
  memcpy(encryptedMap_omp, encryptedMap, sizeof(int)*ENCRYPTEDLEN);

  unsigned int* states;
  states = (unsigned int*)malloc(THREADS*sizeof(unsigned int));

  for (int i=0; i<THREADS; ++i)
    states[i] = i;

  #pragma omp parallel for num_threads(THREADS)
  for (int i=0; i<THREADS; ++i) {
    int idx = omp_get_thread_num();
    int localState = states[idx];

    for (int j=0; j<ALPHABET;++j)
      for (int jj=0; jj<ALPHABET; ++jj)
        scores_omp[j*ALPHABET + jj] = scores[j*ALPHABET + jj];

    int key[KEY_LENGTH];
    for (int j=0; j<KEY_LENGTH; ++j)
      key[j] = j;

    for (int j=0; j<KEY_LENGTH; ++j) {
      int leftLetter = (rand_r(&localState) % KEY_LENGTH);
      int rightLetter = (rand_r(&localState) % KEY_LENGTH);
      int temp = key[leftLetter];
      key[leftLetter] = key[rightLetter];
      key[rightLetter] = temp;
    }

    for (int cycles=0; cycles<CLIMBINGS; ++cycles) {  

      int backupKey[KEY_LENGTH];
      for (int j=0; j<KEY_LENGTH; ++j)
        backupKey[j] = key[j];

      float tempScore = 0.f;

      int branch = (rand_r(&localState) % 100);

      if (branch < HEUR_THRESHOLD_OP1)
      {
        for (int j=0; j<1+(rand_r(&localState) % OP1_HOP); j++) 
        {
          leftLetter = (rand_r(&localState) % KEY_LENGTH);   
          rightLetter = (rand_r(&localState) % KEY_LENGTH); 
          int temp = key[leftLetter];
          key[leftLetter] = key[rightLetter];
          key[rightLetter] = temp;
        }            
      }

      else if (branch < HEUR_THRESHOLD_OP2)
      {
        for (int j=0; j< 1+(rand_r(&localState) % OP2_HOP);j++)
        {
          int blockStart = (rand_r(&localState) % KEY_LENGTH);
          int blockEnd = (rand_r(&localState) % KEY_LENGTH);
          int length = 1+(rand_r(&localState) % (abs((blockStart-blockEnd))-1));
          for (int i=0; i<length; ++i) 
          {
            int leftLetter = (blockStart+i)%KEY_LENGTH;
            int rightLetter = (blockEnd+i)%KEY_LENGTH;
            int temp = key[leftLetter];
            key[leftLetter] = key[rightLetter];
            key[rightLetter] = temp;
          }
        }
      }

      else 
      {
        int l = 1 + (rand_r(&localState) % (KEY_LENGTH-2));
        int f = (rand_r(&localState) % (KEY_LENGTH-1));
        int t = (f+1+(rand_r(&localState) % (KEY_LENGTH-2)));
        t = t % KEY_LENGTH;

        int shiftHelper[KEY_LENGTH];
        for (int j=0; j<KEY_LENGTH; ++j)
          shiftHelper[j] = key[j];

        int t0 = (t-f+KEY_LENGTH) % KEY_LENGTH;
        int n = (t0+l) % KEY_LENGTH;

        for (int j=0; j<n; ++j) 
        {
          int ff = (f+j) % KEY_LENGTH;
          int tt = (((t0+j)%n)+f)%KEY_LENGTH;
          int temp = shiftHelper[ff];
          shiftHelper[ff] = shiftHelper[tt];
          shiftHelper[tt] = temp;
        }        
      }      

      int localDecrypted[ENCRYPTEDLEN];  
      decrypt(encryptedMap_omp, key, localDecrypted);    

      for (int j=0; j<ENCRYPTEDLEN-1; ++j) {
        tempScore += scores_omp[ALPHABET*localDecrypted[j] + localDecrypted[j+1]];
      }

      if (tempScore < 999999.f) {
        for (int j=0; j<ENCRYPTEDLEN; ++j) {
          decrypted_omp[idx*ENCRYPTEDLEN+j] = localDecrypted[j];
        }
      }    

      else 
      {
        for (int j=0; j<KEY_LENGTH; ++j)
          key[j] = backupKey[j];      
      }
    }

    for (int j=0; j<ENCRYPTEDLEN; ++j)
      decrypted_omp[idx*ENCRYPTEDLEN+j] = decrypted_omp[idx*ENCRYPTEDLEN+j];
  }

  int bestCandidate = 0;
  float bestScore = CAP;
  float* scoreHistory = (float*)malloc(B*T*sizeof(float));

  for (int j=0; j<THREADS; ++j)  {
    float currentScore = candidateScore(&decrypted_omp[ENCRYPTEDLEN*j], scores);
    scoreHistory[j] = currentScore;
    if (currentScore < bestScore) {
      bestScore = currentScore;
      bestCandidate = j;
    }    
  }  

  bool pass = verify(&decrypted_omp[ENCRYPTEDLEN*bestCandidate]);
  printf("%s\n", pass ? "PASS" : "FAIL");

  free(scores_omp);
  free(encryptedMap_omp);
  free(decrypted_omp);
  free(states);
  free(scoreHistory);
  return 0;
}
```