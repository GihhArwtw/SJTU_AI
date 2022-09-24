`timescale 1ns/100ps

module detector(
    input wire clk,
    input wire rst_n,
    input wire din,
    output reg detector
);

reg[2:0] state, next_state;

parameter S0 = 0;  // IDLE
parameter S1 = 1;  // 1
parameter S2 = 2;  // 10
parameter S3 = 3;  // 101
parameter S4 = 4;  // 1010
parameter S5 = 5;  // 10100
parameter S6 = 6;  // 101001

always @(posedge clk or negedge rst_n)
begin
    if (!rst_n)
    begin
        state <= S0;   // reset
        next_state <= S0;
        detector <= 0;
    end
end

always @(posedge clk)
begin
    if (rst_n)
    begin
        case ({state,din})
            4'b0000:   next_state = S0;
            4'b0001:   next_state = S1;
            4'b0010:   next_state = S2;
            4'b0011:   next_state = S1;
            4'b0100:   next_state = S0;
            4'b0101:   next_state = S3;
            4'b0110:   next_state = S4;
            4'b0111:   next_state = S1;
            4'b1000:   next_state = S5;
            4'b1001:   next_state = S3;
            4'b1010:   next_state = S0;
            4'b1011:   next_state = S6;
            4'b1100:   next_state = S2;
            4'b1101:   next_state = S1;
            default:   next_state = S0;
        endcase
        state <= next_state;
        detector <= (state==S6);
    end
end


endmodule