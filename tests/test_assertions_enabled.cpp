#include <cassert>
#ifdef NDEBUG
#error "Native qualification assertions must remain enabled in Release builds"
#endif
int main() { int exercised=0; assert(++exercised==1); return exercised==1?0:1; }
