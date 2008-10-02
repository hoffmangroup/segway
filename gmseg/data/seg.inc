#ifndef SEG_INC
#define SEG_INC

%% cardinalities

% data exists or is missing?
#define CARD_NONMISSING 2

% which segment?
#define CARD_SEG 2

%% types

#define TYPE_SEG discrete hidden cardinality CARD_SEG
#define TYPE_DINUCLEOTIDE discrete observed

#endif
