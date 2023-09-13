section .text
global matrixMultiplicationNASM

matrixMultiplicationNASM:
    push rsi
    push rdi
    
    mov rsi, rcx        ; rsi = Q
    mov rdi, rdx        ; rdi = U
    mov ebp, [rsp + 56] ; ebx = j
    mov edx, [rsp + 64] ; edx = n
	
    xorpd xmm0, xmm0 ; temp = 0
    mov eax, r8d        ; l = k

loop_start:
    mov ecx, eax        ; ecx = l
    imul ecx, edx       ; ecx = l * n
    add ecx, ebp        ; ecx = l * n + j
    
    movsd xmm2, qword [rsi + rcx * 8] ; xmm2 = Q[l][j]
    
    mov r11d, r8d       ; rcx = k
    imul r11d, edx      ; rcx = k * n
    add r11d, eax       ; rcx = k * n + l

    mulsd xmm1, qword [rdi + r11 * 8] ; xmm1 = Q[l][j] * U[k][l]
    addsd xmm0, xmm1   ; temp += xmm1

    inc eax             ; ++l
    cmp eax, edx        ; if (l < n)
    jl loop_start

    mov ecx, r9d        ; rcx = i
    imul ecx, edx       ; rcx = i * n
    add ecx, r8d        ; rcx = i * n + k
    movsd xmm4, qword [rsi + rcx * 8] ; xmm4 = Q[i][k]

    mulsd xmm0, xmm4    ; temp *= Q[i][k]

    pop rdi
    pop rsi
    ret

    

















