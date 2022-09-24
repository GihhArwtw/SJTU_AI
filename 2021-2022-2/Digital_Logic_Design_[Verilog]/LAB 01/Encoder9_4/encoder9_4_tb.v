`timescale 1ns/100ps

module encoder_9_4_tb;

reg [8:0] a_in, a_gen;
wire[3:0] y_out;


encoder9_4 ENCODER(
    .a(a_in),
    .y(y_out)
);

initial fork
    a_gen = 0;
    a_in  = 0;
join

always #20        // We also test our encoder on invalid inputs.
begin
    // a_in = a_in + 1;   // If we need to test all possible inputs in order.
    
    a_gen = a_gen << 1;
    if (a_gen==0)
        a_gen = 1;
    
    a_in = a_gen;
    #5 a_in = a_gen + {$random} % a_gen;   // We test three random inputs.
    #5 a_in = a_gen + {$random} % a_gen;
    #5 a_in = a_gen + {$random} % a_gen;

    //ran=$random % 60;
end

endmodule