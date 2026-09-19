%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                Emmanouil Thomas Chatzakis                  %
%                2021030061                                  %
%                                                            %
%   Wireless Communications - Exercise 4                     %
%   Runs the four parts and keeps every printed result in    %
%   results_parts_log.txt                                    %
%                                                            %
%   Each part starts with "clear", so the output of every    %
%   script is captured with evalc and written to the log     %
%   immediately after it returns.                            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
close all;
clc;

txt = evalc('Ex_4_part1');
fid = fopen('results_parts_log.txt', 'w');
fprintf(fid, '===== PART 1 : ideal channel =====\n%s', txt);
fclose(fid);

txt = evalc('Ex_4_part2');
fid = fopen('results_parts_log.txt', 'a');
fprintf(fid, '\n===== PART 2 : ideal channel with CFO =====\n%s', txt);
fclose(fid);

txt = evalc('Ex_4_part3');
fid = fopen('results_parts_log.txt', 'a');
fprintf(fid, '\n===== PART 3 : non ideal channel =====\n%s', txt);
fclose(fid);

txt = evalc('Ex_4_part4');
fid = fopen('results_parts_log.txt', 'a');
fprintf(fid, '\n===== PART 4 : non ideal channel with CFO =====\n%s', txt);
fclose(fid);

type('results_parts_log.txt');
