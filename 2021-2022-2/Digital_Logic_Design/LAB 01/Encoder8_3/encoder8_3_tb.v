`timescale 1ns/100ps

module encoder8_3_tb;

reg [7:0] a_in;
wire[2:0] y_out;
reg [1:0] count;


encoder8_3 ENCODER(
    .a(a_in),
    .y(y_out)
);

initial fork
    a_in = 0;
    count = 0;
join

always #5        // We also test our encoder on invalid inputs.
begin
    if (a_in==0)
    begin
        a_in = {$random}%256;          // We test three random inputs.
        count = 1;
    end
    else if ((count>0) && (count<3))   // We test three random inputs.
    begin
        a_in = {$random}%256;
        count = count+1;
    end
    else if (count==3)
    begin
        a_in = 8'b00000001;
        count = 0;
    end
    else
        a_in = a_in<<1;
end

endmodule