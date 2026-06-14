Here's the equivalent serial C++ code for the given CUDA code:

```cpp
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>

#define rot(x,k) (((x)<<(k)) | ((x)>>(32-(k))))

#define mix(a,b,c) \
{ \
  a -= c;  a ^= rot(c, 4);  c += b; \
  b -= a;  b ^= rot(a, 6);  a += c; \
  c -= b;  c ^= rot(b, 8);  b += a; \
  a -= c;  a ^= rot(c,16);  c += b; \
  b -= a;  b ^= rot(a,19);  a += c; \
  c -= b;  c ^= rot(b, 4);  b += a; \
}

#define final(a,b,c) \
{ \
  c ^= b; c -= rot(b,14); \
  a ^= c; a -= rot(c,11); \
  b ^= a; b -= rot(a,25); \
  c ^= b; c -= rot(b,16); \
  a ^= c; a -= rot(c,4);  \
  b ^= a; b -= rot(a,14); \
  c ^= b; c -= rot(b,24); \
}

unsigned int mixRemainder(unsigned int a, 
    unsigned int b, 
    unsigned int c, 
    unsigned int k0,
    unsigned int k1,
    unsigned int k2,
    unsigned int length ) 
{
  switch(length)
  {
    case 12: c+=k2; b+=k1; a+=k0; break;
    case 11: c+=k2&0xffffff; b+=k1; a+=k0; break;
    case 10: c+=k2&0xffff; b+=k1; a+=k0; break;
    case 9 : c+=k2&0xff; b+=k1; a+=k0; break;
    case 8 : b+=k1; a+=k0; break;
    case 7 : b+=k1&0xffffff; a+=k0; break;
    case 6 : b+=k1&0xffff; a+=k0; break;
    case 5 : b+=k1&0xff; a+=k0; break;
    case 4 : a+=k0; break;
    case 3 : a+=k0&0xffffff; break;
    case 2 : a+=k0&0xffff; break;
    case 1 : a+=k0&0xff; break;
    case 0 : return c;              
  }

  final(a,b,c);
  return c;
}

unsigned int hashlittle(const void *key, size_t length, unsigned int initval)
{
  unsigned int a, b, c;                                          

  a = b = c = 0xdeadbeef + ((unsigned int)length) + initval;

  const unsigned int *k = (const unsigned int *)key;         

  while (length > 12)
  {
    a += k[0];
    b += k[1];
    c += k[2];
    mix(a, b, c);
    length -= 12;
    k += 3;
  }

  switch(length)
  {
    case 12: c+=k[2]; b+=k[1]; a+=k[0]; break;
    case 11: c+=k[2]&0xffffff; b+=k[1]; a+=k[0]; break;
    case 10: c+=k[2]&0xffff; b+=k[1]; a+=k[0]; break;
    case 9 : c+=k[2]&0xff; b+=k[1]; a+=k[0]; break;
    case 8 : b+=k[1]; a+=k[0]; break;
    case 7 : b+=k[1]&0xffffff; a+=k[0]; break;
    case 6 : b+=k[1]&0xffff; a+=k[0]; break;
    case 5 : b+=k[1]&0xff; a+=k[0]; break;
    case 4 : a+=k[0]; break;
    case 3 : a+=k[0]&0xffffff; break;
    case 2 : a+=k[0]&0xffff; break;
    case 1 : a+=k[0]&0xff; break;
    case 0 : return c;              
  }

  final(a, b, c);
  return c;
}

int main(int argc, char** argv) {

  if (argc != 4) {
    std::cerr << "Usage: " << argv[0] << " <block size> <number of strings> <repeat>" << std::endl;
    return 1;
  }

  int block_size = std::atoi(argv[1]);  
  unsigned long N = std::atol(argv[2]); 
  int repeat = std::atoi(argv[3]);

  const char* str = "Four score and seven years ago";
  unsigned int c = hashlittle(str, 30, 1);
  std::cout << "input string: " << str << " hash is " << std::hex << c << std::endl;   

  std::vector<unsigned int> keys(N * 16);
  std::vector<unsigned int> lens(N);
  std::vector<unsigned int> initvals(N);
  std::vector<unsigned int> out(N);

  srand(2);
  char src[64];
  memcpy(src, str, 64);
  for (unsigned long i = 0; i < N; i++) {
    memcpy((unsigned char*)&keys[i * 16], src, 64);
    lens[i] = rand() % 61;
    initvals[i] = i % 2;
  }

  for (int n = 0; n < repeat; ++n) {
    for (unsigned long i = 0; i < N; ++i) {
      unsigned int length = lens[i];
      const unsigned int initval = initvals[i];
      const unsigned int *k = &keys[i * 16];  
      unsigned int a, b, c; 
      unsigned int r0, r1, r2;
      a = b = c = 0xdeadbeef + length + initval;

      while (length > 12) {
        a += k[0];
        b += k[1];
        c += k[2];
        mix(a, b, c);
        length -= 12;
        k += 3;
      }
      r0 = k[0];
      r1 = k[1];
      r2 = k[2];

      out[i] = mixRemainder(a, b, c, r0, r1, r2, length);
    }
  }

  std::cout << "Verify the results computed on the device.." << std::endl;
  bool error = false;
  for (unsigned long i = 0; i < N; i++) {
    c = hashlittle(&keys[i * 16], lens[i], initvals[i]);
    if (out[i] != c) {
      std::cerr << "Error: at " << i << " gpu hash is " << std::hex << out[i] << "  cpu hash is " << c << std::endl;
      error = true;
      break;
    }
  }

  std::cout << (error ? "FAIL" : "PASS") << std::endl;

  return 0;
}
```

This code performs the same operations as the original CUDA code but runs them sequentially on a single thread. The `kernel` function has been replaced with nested loops that iterate over the data, performing the same computations. The memory allocation and deallocation have also been simplified since there is no need for GPU-specific memory management in this serial version.