#pragma once

#include <stdint.h>
#include <torch/extension.h>
#include <torch/torch.h>

#define DIV_UP(x, y) ((x) + (y) - 1) / (y)
#define OFFSET(row, col, ld) ((row) * (ld) + (col))

// Define rope type
enum ROPE_TYPE {
  NO_ROPE = 0,    // do not use rope
  FULL_ROPE = 1,  // use rope for all headdim
  HALF_ROPE = 2   // use rope for half of headdim
};

// Define which dim the sequence length is
enum SEQ_DIM_TYPE {
  FIRST = 0,   // [seqlen, ...]
  SECOND = 1,  // [..., seqlen, ...]
};

// Define attention mask type
enum MASK_TYPE {
  NO_MASK = 0,    // do not use mask
  ALIBI_MASK = 1  // use alibi mask
};

// Define whether the token index is aligned for different token
enum FREQ_ALIGNED {
  NO = 0,
  YES = 1
};