onerror {resume}
quietly WaveActivateNextPane {} 0
add wave -noupdate /Bayer2RGB_tb/Bayer2RGB_inst/clk
add wave -noupdate /Bayer2RGB_tb/Bayer2RGB_inst/rst_n
add wave -noupdate -radix hexadecimal /Bayer2RGB_tb/Bayer2RGB_inst/data_in
add wave -noupdate /Bayer2RGB_tb/Bayer2RGB_inst/cen
add wave -noupdate -radix unsigned /Bayer2RGB_tb/Bayer2RGB_inst/addr
add wave -noupdate /Bayer2RGB_tb/Bayer2RGB_inst/O_RGB_data_valid
add wave -noupdate -radix hexadecimal /Bayer2RGB_tb/Bayer2RGB_inst/O_RGB_data_R
add wave -noupdate -radix hexadecimal /Bayer2RGB_tb/Bayer2RGB_inst/O_RGB_data_G
add wave -noupdate -radix hexadecimal /Bayer2RGB_tb/Bayer2RGB_inst/O_RGB_data_B
add wave -noupdate /Bayer2RGB_tb/Bayer2RGB_inst/mask
add wave -noupdate /Bayer2RGB_tb/Bayer2RGB_inst/pattern
add wave -noupdate -radix unsigned /Bayer2RGB_tb/Bayer2RGB_inst/line
add wave -noupdate -radix unsigned /Bayer2RGB_tb/Bayer2RGB_inst/count
add wave -noupdate /Bayer2RGB_tb/Bayer2RGB_inst/fetch_count
add wave -noupdate -radix hexadecimal /Bayer2RGB_tb/Bayer2RGB_inst/pixel_r
add wave -noupdate -radix hexadecimal /Bayer2RGB_tb/Bayer2RGB_inst/pixel_g
add wave -noupdate -radix hexadecimal /Bayer2RGB_tb/Bayer2RGB_inst/pixel_b
add wave -noupdate /Bayer2RGB_tb/Bayer2RGB_inst/hold
TreeUpdate [SetDefaultTree]
WaveRestoreCursors {{Cursor 1} {1499998766 ps} 0}
quietly wave cursor active 1
configure wave -namecolwidth 150
configure wave -valuecolwidth 100
configure wave -justifyvalue left
configure wave -signalnamewidth 0
configure wave -snapdistance 10
configure wave -datasetprefix 0
configure wave -rowmargin 4
configure wave -childrowmargin 2
configure wave -gridoffset 0
configure wave -gridperiod 1
configure wave -griddelta 40
configure wave -timeline 0
configure wave -timelineunits ps
update
WaveRestoreZoom {1499998696 ps} {1500000069 ps}
