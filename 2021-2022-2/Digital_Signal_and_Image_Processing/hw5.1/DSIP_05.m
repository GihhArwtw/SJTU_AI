x = imread("roman.jpg");


% TASK 01
x_r = x(:,:,1);
x_r_uni = histeq(x_r);

fig = figure;
set(fig,'position',[0,0,1440,900]);
imshow(x_r);

fig = figure;
set(fig,'position',[0,0,1440,900]);
imshow(x_r_uni);

% TASK 02
[exp, bins] = imhist(uint8(exprnd(100,10000)),64);
x_r_exp = histeq(x_r,exp);
[norm,bins] = imhist(uint8(normrnd(100,75,10000)),64);
x_r_norm = histeq(x_r,norm);

fig = figure;
set(fig,'position',[0,0,1440,900]);
imshow(x_r_exp);

fig = figure;
set(fig,'position',[0,0,1440,900]);
imshow(x_r_norm);


% TASK 03
[ryl, bins] = imhist(uint8(random('Rayleigh',75,10000)),64);
x_r_ryl = histeq(x_r,ryl);

fig = figure;
set(fig,'position',[0,0,1440,900]);
imshow(x_r_ryl);

fig = figure;
sgtitle('Histagram Equation Matching Different Distributions','Fontsize',12)
set(fig,'position',[0.1,0.1,1500,500]); 
subplot(2,5,1), imshow(x_r);
title('Original Red Channel');
subplot(2,5,6), imhist(x_r);
ylim([0,60000]);

subplot(2,5,2), imshow(x_r_uni);
title('HE - Uniform Distribution');
subplot(2,5,7), imhist(x_r_uni);
ylim([0,60000]);

subplot(2,5,3), imshow(x_r_exp);
title('HE - Exponential Distribution');
subplot(2,5,8), imhist(x_r_exp);
ylim([0,100000]);

subplot(2,5,4), imshow(x_r_norm);
title('HE - Gaussian Distribution');
subplot(2,5,9), imhist(x_r_norm);
ylim([0,100000]);

subplot(2,5,5), imshow(x_r_ryl);
title('HE - Rayleigh Distribution');
subplot(2,5,10), imhist(x_r_ryl);
ylim([0,100000]);

saveas(fig, "TASK01-03.bmp");
saveas(fig, "TASK01-03.svg");


% TASK 04
x_g = x(:,:,2);
x_b = x(:,:,3);
x_g_uni = histeq(x_g);
x_b_uni = histeq(x_b);
x_eq = cat(3,x_r_uni, x_g_uni, x_b_uni);

fig = figure;
set(fig,'position',[0,0,1440,900]);
imshow(x_eq);


% TASK 05
[ryl, bins] = imhist(uint8(random('Rayleigh',100,10000)),64);
x_r_ryl = histeq(x_r,ryl);
x_g_ryl = histeq(x_g,ryl);
x_b_ryl = histeq(x_b,ryl);
x_ryl = cat(3,x_r_ryl,x_g_ryl,x_b_ryl);

x_r_clahe = adapthisteq(x_r);
x_g_clahe = adapthisteq(x_g);
x_b_clahe = adapthisteq(x_b);
x_clahe = cat(3,x_r_clahe,x_g_clahe,x_b_clahe);

fig = figure;
set(fig,'position',[0,0,1440,900]);
imshow(x_ryl);

fig = figure;
set(fig,'position',[0,0,1440,900]);
imshow(x_clahe);

fig = figure;
set(fig,'position',[0.1,0.1,2000,500]); 
subplot(4,4,1), imshow(x);
title('Original Image');
subplot(4,4,2), imhist(x_r);
ylim([0,60000]);
title('R');
subplot(4,4,3), imhist(x_g);
ylim([0,60000]);
title('G');
subplot(4,4,4), imhist(x_b);
ylim([0,60000]);
title('B');

subplot(4,4,5), imshow(x_eq);
title('Histogram Equation');
subplot(4,4,6), imhist(x_r_uni);
ylim([0,100000]);
title('R');
subplot(4,4,7), imhist(x_g_uni);
ylim([0,100000]);
title('G');
subplot(4,4,8), imhist(x_b_uni);
ylim([0,100000]);
title('B');

subplot(4,4,9), imshow(x_ryl);
title('HE Matching Ryl');
subplot(4,4,10), imhist(x_r_ryl);
ylim([0,100000]);
title('R');
subplot(4,4,11), imhist(x_g_ryl);
ylim([0,100000]);
title('G');
subplot(4,4,12), imhist(x_b_ryl);
ylim([0,100000]);
title('B');

subplot(4,4,13), imshow(x_clahe);
title('HE Matching Exp');
subplot(4,4,14), imhist(x_r_clahe);
ylim([0,60000]);
title('R');
subplot(4,4,15), imhist(x_g_clahe);
ylim([0,60000]);
title('G');
subplot(4,4,16), imhist(x_b_clahe);
ylim([0,60000]);
title('B');

saveas(fig,"TASK04-05.bmp");
saveas(fig,"TASK04-05.svg");