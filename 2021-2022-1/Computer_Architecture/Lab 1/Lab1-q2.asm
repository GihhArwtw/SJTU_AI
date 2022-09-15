	.ORIG	x3000
	LDI	R0, SEQ
	LDI	R1, STP		; is STP>0 ?
	BRnz	INVALID

	LD	R2, NEGSXTN	; is STP<16?
	ADD	R2, R1, R2
	BRzp	INVALID

	AND	R4, R4, #0	; the current bit after the shift
	ADD	R4, R4, #1
	AND	R5, R5, #0	; the final result
	STI	R5, FLAG	; is valid (0)
	AND	R2, R2, #0	; mask
	ADD	R2, R2, #1

DOUBLE	ADD	R2, R2, R2
	ADD	R1, R1, #-1
	BRz	MASK
	BRnzp	DOUBLE

MASK	AND	R3, R0, R2
	BRz	SKIP
	ADD	R5, R5, R4
SKIP	ADD	R4, R4, R4
	ADD	R2, R2, R2
	BRz	END
	BRnzp	MASK

END	STI	R5, RES
	HALT
	

INVALID	AND	R5, R5, #0
	STI	R5, RES
	ADD	R5, R5, #1
	STI	R5, FLAG
	HALT
	

SEQ	.FILL	x3100
STP	.FILL	x3101
RES	.FILL	x3102
FLAG	.FILL	x3103
NEGSXTN	.FILL	#-16

	.END