      struct Uniforms { output_size:u32, pads:array<vec4<i32>, 2>, constant_value:f16, constant_vec2:vec2<f16>, constant_vec3:vec3<f16>, constant_vec4:vec4<f16>,  constant_vec5:array<mat2x4<f16>, 1>,  constant_vec9:array<mat2x4<f16>, 2>, x_shape:vec4<u32>, x_strides:vec4<u32>, output_shape:vec4<u32>, output_strides:vec4<u32> };
                                                                                                                                                                   ^^^^^^^^^^^^^

:3:7 note: see layout of struct:
/*             align(16) size(192) */ struct Uniforms {
/* offset(  0) align( 4) size(  4) */   output_size : u32; // 0 + 4
/* offset(  p4) align( 1) size( p12) */   // -- implicit field alignment padding --;
/* offset( 16) align(16) size( 32) */   pads : array<vec4<i32>, 2>; // 16+32
/* offset( 48) align( 2) size(  2) */   constant_value : f16; // 48+2
/* offset( p50) align( 1) size(  p2) */   // -- implicit field alignment padding --;
/* offset( 52) align( 4) size(  4) */   constant_vec2 : vec2<f16>; // 52+4
/* offset( 56) align( 8) size(  6) */   constant_vec3 : vec3<f16>; // 56+6
/* offset( p62) align( 1) size(  p2) */   // -- implicit field alignment padding --;
/* offset( 64) align( 8) size(  8) */   constant_vec4 : vec4<f16>; //64 + 8
/* offset( 72) align( 8) size( 16) */   constant_vec5 : array<mat2x4<f16>, 1>;
/* offset( 88) align( 8) size( 32) */   constant_vec9 : array<mat2x4<f16>, 2>;
/* offset(p120) align( 1) size(  p8) */   // -- implicit field alignment padding --;
/* offset(128) align(16) size( 16) */   x_shape : vec4<u32>;
/* offset(144) align(16) size( 16) */   x_strides : vec4<u32>;
/* offset(160) align(16) size( 16) */   output_shape : vec4<u32>;
/* offset(176) align(16) size( 16) */   output_strides : vec4<u32>;
/*                                 */ };


:3:7 note: see layout of struct:
/*             align(16) size(224) */ struct Uniforms {
/* offset(  0) align( 4) size(  4) */   output_size : u32;
/* offset(  4) align( 1) size( 12) */   // -- implicit field alignment padding --;
/* offset( 16) align(16) size( 32) */   pads : array<vec4<i32>, 2>;
/* offset( 48) align( 2) size(  2) */   constant_value : f16;
/* offset( 50) align( 1) size(  2) */   // -- implicit field alignment padding --;
/* offset( 52) align( 4) size(  4) */   constant_vec2 : vec2<f16>;
/* offset( 56) align( 8) size(  6) */   constant_vec3 : vec3<f16>;
/* offset( 62) align( 1) size(  2) */   // -- implicit field alignment padding --;
/* offset( 64) align( 8) size(  8) */   constant_vec4 : vec4<f16>;
/* offset( 72) align( 1) size(  8) */   // -- implicit field alignment padding --;
/* offset( 80) align(16) size( 16) */   constant_vec5 : array<mat2x4<f16>, 1>;
/* offset( 96) align(16) size( 32) */   constant_vec9 : array<mat2x4<f16>, 2>;
/* offset(128) align( 8) size(  8) */   constant_vec4_1 : vec4<f16>;
/* offset(136) align( 8) size( 16) */   constant_vec6 : array<mat2x4<f16>, 1>;
/* offset(152) align( 1) size(  8) */   // -- implicit field alignment padding --;
/* offset(160) align(16) size( 16) */   x_shape : vec4<u32>;
/* offset(176) align(16) size( 16) */   x_strides : vec4<u32>;
/* offset(192) align(16) size( 16) */   output_shape : vec4<u32>;
/* offset(208) align(16) size( 16) */   output_strides : vec4<u32>;

oo1 currentOffset=0, data.length=1, baseAlignment=4, v.type= uint32
oo2 currentOffset=0
oo3_2 currentOffset=4, size = 4
oo1 currentOffset=4, data.length=8, baseAlignment=16, v.type= uint32
oo2 currentOffset=16
oo3_2 currentOffset=48, size = 32
oo1 currentOffset=48, data.length=1, baseAlignment=2, v.type= float16
oo2 currentOffset=48
oo3_1 currentOffset=50, size = 2
oo1 currentOffset=50, data.length=2, baseAlignment=4, v.type= float16
oo2 currentOffset=52
oo3_1 currentOffset=56, size = 4
oo1 currentOffset=56, data.length=3, baseAlignment=8, v.type= float16
oo2 currentOffset=56
oo3_1 currentOffset=62, size = 6
oo1 currentOffset=62, data.length=4, baseAlignment=8, v.type= float16
oo2 currentOffset=64
oo3_1 currentOffset=72, size = 8
oo1 currentOffset=72, data.length=5, baseAlignment=16, v.type= float16
oo2 currentOffset=80
oo3_1 currentOffset=96, size = 16
oo1 currentOffset=96, data.length=9, baseAlignment=16, v.type= float16
oo2 currentOffset=96
oo3_1 currentOffset=128, size = 32
oo1 currentOffset=128, data.length=4, baseAlignment=16, v.type= uint32
oo2 currentOffset=128
oo3_2 currentOffset=144, size = 16
oo1 currentOffset=144, data.length=4, baseAlignment=16, v.type= uint32
oo2 currentOffset=144
oo3_2 currentOffset=160, size = 16
oo1 currentOffset=160, data.length=4, baseAlignment=16, v.type= uint32
oo2 currentOffset=160
oo3_2 currentOffset=176, size = 16
oo1 currentOffset=176, data.length=4, baseAlignment=16, v.type= uint32
oo2 currentOffset=176
oo3_2 currentOffset=192, size = 16
oo3_3 currentOffset=192
program-manager.ts:149 enable f16;
