onerror {resume}
quietly WaveActivateNextPane {} 0
add wave -noupdate /countdown_tb/CLK
add wave -noupdate /countdown_tb/LD
add wave -noupdate -radix unsigned /countdown_tb/DATA
add wave -noupdate -radix unsigned -childformat {{{/countdown_tb/Counter/counter[9]} -radix unsigned} {{/countdown_tb/Counter/counter[8]} -radix unsigned} {{/countdown_tb/Counter/counter[7]} -radix unsigned} {{/countdown_tb/Counter/counter[6]} -radix unsigned} {{/countdown_tb/Counter/counter[5]} -radix unsigned} {{/countdown_tb/Counter/counter[4]} -radix unsigned} {{/countdown_tb/Counter/counter[3]} -radix unsigned} {{/countdown_tb/Counter/counter[2]} -radix unsigned} {{/countdown_tb/Counter/counter[1]} -radix unsigned} {{/countdown_tb/Counter/counter[0]} -radix unsigned}} -subitemconfig {{/countdown_tb/Counter/counter[9]} {-radix unsigned} {/countdown_tb/Counter/counter[8]} {-radix unsigned} {/countdown_tb/Counter/counter[7]} {-radix unsigned} {/countdown_tb/Counter/counter[6]} {-radix unsigned} {/countdown_tb/Counter/counter[5]} {-radix unsigned} {/countdown_tb/Counter/counter[4]} {-radix unsigned} {/countdown_tb/Counter/counter[3]} {-radix unsigned} {/countdown_tb/Counter/counter[2]} {-radix unsigned} {/countdown_tb/Counter/counter[1]} {-radix unsigned} {/countdown_tb/Counter/counter[0]} {-radix unsigned}} /countdown_tb/Counter/counter
add wave -noupdate /countdown_tb/OUT
TreeUpdate [SetDefaultTree]
WaveRestoreCursors {{Cursor 1} {73300 ps} 0}
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
WaveRestoreZoom {18900 ps} {311 ns}
