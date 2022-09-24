`timescale 1ns/100ps

module countdown(
    input wire clk,
    input wire load,
    input wire[9:0] data_load,
    output reg tc
    );

reg[9:0] counter;

always @(posedge clk)
begin
    if (load==1'b1)                     // if LOAD signal is valid
        counter = data_load;
    else
    begin
        if (counter>0)
            counter = counter-1;
    end

    if (counter==0)                     // if it is already terminal count. 
        tc = 1;
    else
        tc = 0;
end


endmodule