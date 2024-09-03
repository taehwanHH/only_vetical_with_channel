clear all; 
% close all;
% .mat 파일 로드
data_path = '2024-08-30/5/data.mat';
data = load(data_path);

window_size = 50;
lifting_time = data.stable_lifting_time;
avg_lifting_time = movmean(lifting_time, window_size);
box_pos = data.box_z_pos;
avg_box_pos = movmean(box_pos, window_size);

figure(1);
plot(avg_lifting_time,'r','LineWidth',1.5);
ylim([0 501]);
grid on;
xlabel('Episode','FontSize',14,'Interpreter','latex');
ylabel('Time holding the object stable','FontSize',14,'Interpreter','latex');
title('Time holding the object stable over episodes','FontSize',14,'Interpreter','latex');

figure(2);
plot(avg_box_pos,'b','LineWidth',1.5);
grid on;
ylim([0,2.5]);
xlabel('Episode','FontSize',14,'Interpreter','latex');
ylabel('Box z-pos','FontSize',14,'Interpreter','latex');
title('Time holding the object stable over episodes','FontSize',14,'Interpreter','latex');