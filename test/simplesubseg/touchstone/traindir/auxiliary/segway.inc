#ifndef SEG_INC
#define SEG_INC

% not used yet:
% #define USE_MFSDG

%% component types
%% enumerated in tksrc/GMTK_Component.h:ComponentType
#define COMPONENT_TYPE_DIAG_GAUSSIAN 0
#define COMPONENT_TYPE_GAMMA 3
#define COMPONENT_TYPE_MISSING_FEATURE_SCALED_DIAG_GAUSSIAN 5

%% cardinalities

% any boolean (such as alwaysTrue)
#define CARD_BOOLEAN 2

% data exists or is missing?

#ifdef USE_MFSDG
% only continues to exist for dinucleotide
#define CARD_PRESENCE CARD_BOOLEAN
#else /* USE_MFSDG */
#define CARD_PRESENCE 2
#endif /* USE_MFSDG */

% transition or not?
#define CARD_SEGTRANSITION 3

% maximum number of frames
#ifndef CARD_FRAMEINDEX
#define CARD_FRAMEINDEX 2000000
#endif

% allow transition or not?
#define CARD_RULER CARD_BOOLEAN

% others
#ifndef CARD_SEGCOUNTDOWN
#define CARD_SEGCOUNTDOWN 2
#endif

% want to allow overriding this
#ifndef CARD_SEG
#define CARD_SEG 2
#endif

#ifndef CARD_SUBSEG
#define CARD_SUBSEG 2
#endif

% supervision label
#ifndef CARD_SUPERVISIONLABEL
#define CARD_SUPERVISIONLABEL 3
#endif

% supervision label extension
#ifndef SUPERVISIONLABEL_RANGE_SIZE
#define SUPERVISIONLABEL_RANGE_SIZE 1
#endif

% AA .. TT
#define CARD_DINUCLEOTIDE 16

%% types

#define TYPE_SEGCOUNTDOWN discrete hidden cardinality CARD_SEGCOUNTDOWN
#define TYPE_SEG discrete hidden cardinality CARD_SEG
#define TYPE_SUBSEG discrete hidden cardinality CARD_SUBSEG

%% conditionalparents

#define CONDITIONALPARENTS_NIL_CONTINUOUS nil using mixture("internal:UnityScore")
#define CONDITIONALPARENTS_NIL_DISCRETE nil using DenseCPT("internal:UnityScore")
#define CONDITIONALPARENTS_PRESENCE CONDITIONALPARENTS_NIL_DISCRETE

#if CARD_SUBSEG <= 1
#define CONDITIONALPARENTS_OBS seg(0)
#define MAPPING_OBS mapping("internal:copyParent")
#else /* CARD_SUBSEG > 1 */
#define CONDITIONALPARENTS_OBS seg(0), subseg(0)
#define MAPPING_OBS mapping("map_seg_subseg_obs")
#endif /* CARD_SUBSEG == 1 */

%% misc
#define RULER_SCALE 10
#define SUPERVISION_LABEL_OFFSET 1

#ifndef SEGTRANSITION_WEIGHT_SCALE
#define SEGTRANSITION_WEIGHT_SCALE 1.0
#endif /* SEGTRANSITION_WEIGHT_SCALE */

#endif
