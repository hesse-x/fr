#pragma once
namespace mapped_group {
struct TracePermutation {
    int kind,first;__m256i indices;
    explicit TracePermutation(uint32_t code):kind(code>>16),first(kind==1?int((code>>12)&15):int(code&15)){
        if(kind==2)indices=_mm256_setr_epi64x(code&15,(code>>4)&15,(code>>8)&15,(code>>12)&15);
    }
};
inline mapped_simd::Vec packed_trace(const double* field,int components,int slot,const TracePermutation& map,int c){
    const double* base=field+(int64_t(slot)*components+c)*16;
    if(map.kind==0)return simd::load(base+map.first);
    if(map.kind==1)return mapped_simd::Vec(_mm256_permute4x64_pd(simd::load(base+map.first).value,0x1b));
    return mapped_simd::Vec(_mm256_i64gather_pd(base,map.indices,8));
}
}
