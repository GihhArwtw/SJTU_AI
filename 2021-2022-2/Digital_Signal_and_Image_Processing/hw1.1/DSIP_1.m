x = imread("baboon.bmp");
size = 516;

% Blurring the image.
ConvKernel= ones(5,5) * 0.04;
y = conv2(x, ConvKernel); %y = conv2(x, ConvKernel, 'same');

fig = figure,
subplot(1,3,1), imshow(x,[min(min(x)),max(max(x))]);
title('original image','FontSize',12);
subplot(1,3,2), imshow(y,[min(min(y)),max(max(y))]);
title('blurred image','FontSize',12);
x_1 = direct_inv(y, ConvKernel, size);
x_1 = x_1(1:512,1:512);
subplot(1,3,3), imshow(x_1,[min(min(x_1)),max(max(x_1))]);
title('restored image','FontSize',12);
set(fig,'position',[0.1,0.1,1000,390]); 
saveas(fig, "step01_baboon_blurred.bmp");


% Noising the image.
y_q = cat(3,y,y,y);

fig = figure,
for i = 1:3
    y_q(:,:,i) = awgn(y, (4-i)*10, 'measured');
    subplot(1,3,i),imshow(y_q(:,:,i),[min(min(y_q(:,:,i))),max(max(y_q(:,:,i)))]);
    title_content = ['Blurred and noised in ', num2str(40-10*i),'dB.'];
    title(title_content,'Fontsize',12);
    %file_name = ['baboon_noise', num2str(40-10*i), 'dB.bmp'];
    %imwrite(uint(y_q(:,:,i)),file_name);
end
set(fig,'position',[0.1,0.1,1000,390]); 
saveas(fig, "step02_baboon_noise.bmp");

% Compare different noising methods.

fig = figure,
subplot(1,3,1),imshow(y_q(:,:,2),[min(min(y_q(:,:,2))),max(max(y_q(:,:,2)))]);
title('Gaussian','FontSize',12);
y_1 = imnoise(rescale(y,0,1), 'poisson');
subplot(1,3,2),imshow(y_1);
title('Poisson','FontSize',12);
y_1 = imnoise(rescale(y,0,1), 'salt & pepper');
subplot(1,3,3),imshow(y_1);
title('Salt & Pepper','FontSize',12);
sgtitle('Comparison of Different Noises');
set(fig,'position',[0.1,0.1,1200,390]);  
saveas(fig, "baboon_noise_diff.bmp");

% Deblurring By Directly Inverting the Filter

fig = figure,
for i = 1:3
    x = direct_inv(y_q(:,:,i), ConvKernel, size);
    subplot(1,3,i),imshow(x,[min(min(x)),max(max(x))]);
    title_content= [num2str(40-10*i),'dB-Noised'];
    title(title_content,'FontSize',12);
end
sgtitle('Restored By Directly Dividing the Filter in the Frequency Domain');
set(fig,'position',[0.1,0.1,1200,390]);  
saveas(fig, "step03.2_baboon_direct_inv.bmp");

% Print the fft of original image and noised images.

fig = figure,
f = log(abs(fftshift(fft2(x))));
subplot(2,3,1), imshow(rescale(f));
title('Original Image','FontSize',12);

f = log(abs(fftshift(fft2(y))));
subplot(2,3,2), imshow(rescale(f));
title('Blurred Image','FontSize',12);

for i = 1:3
    f = log(abs(fftshift(fft2(y_q(:,:,i)))));
    subplot(2,3,3+i), imshow(rescale(f));
    title_content= [num2str(40-10*i),'dB-Noised'];
    title(title_content,'FontSize',12);
end
sgtitle('The FFT of Original Image, Blurred and Noised Images');
set(fig,'position',[0.1,0.1,900,700]);
saveas(fig, "step03.1_baboon_fft2.bmp");

% Deblurring and Denoising Using Wiener Filter.

fig = figure,
for i = 1:3
    x = deconvwnr(y_q(:,:,i), ConvKernel, 40-i*10);
    subplot(1,3,i),imshow(x,[min(min(x)),max(max(x))]);
    title_content= [num2str(40-10*i),'dB-Noised'];
    title(title_content,'FontSize',12);
end
sgtitle('Restored Using Wiener Filter.');
set(fig,'position',[0.1,0.1,1200,390]);  
saveas(fig, "step03.3_baboon_deconv_wnr.bmp");

% Deblurring and Denoising By Lucy-Richardson Method.

fig = figure,
for i = 1:3
    x = deconvlucy(y_q(:,:,i), ConvKernel, 40-i*10);
    subplot(1,3,i),imshow(x,[min(min(x)),max(max(x))]);
    title_content= [num2str(40-10*i),'dB-Noised'];
    title(title_content,'FontSize',12);
end
sgtitle('Restored By Lucy-Richardson Method.');
set(fig,'position',[0.1,0.1,1200,390]);  
saveas(fig, "step03.4_baboon_deconv_lucy.bmp");


%%%%%%%%%%%%%%%%%%%%%%%

function [x] = direct_inv( y, ConvKernel, size)
Y = fft2(y);
C = fft2(ConvKernel,size,size);
X = Y ./ C;
x = ifft2(X,size,size);
end