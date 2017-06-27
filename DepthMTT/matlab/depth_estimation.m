clear;
close all;

% =========================================================================
data_base_path = '../data/MTT_S_01';
depth_image_fig_num = 100;
center_region_ratio = 0.4;
depth_foreground_window_size = 30;
depth_face_window_size = 10;
% =========================================================================

% color preset
NUM_COLORS = 400;
COLORS = distinguishable_colors(NUM_COLORS);
tempColor = COLORS(end, :); % move black color to the end
COLORS(end, :) = COLORS(4, :);
COLORS(4, :) = tempColor;

% main loop
file_list = dir(sprintf('%s/*.png', data_base_path));
file_names = sort({file_list(:).name});
for fIdx = 1:length(file_names)    
    % read image
    depth_image = imread(fullfile(data_base_path, file_names{fIdx}));
    [img_H, img_W, ~] = size(depth_image);
    
    % read box info
    num_objects = 0;
    [~, name, ext] = fileparts(file_names{fIdx});
    gt_file_path = fullfile(data_base_path, [name '.txt']);
    if exist(gt_file_path, 'file')
        fid = fopen(gt_file_path);
        num_objects = str2double(fgetl(fid));
        read_data = fscanf(fid, '%d %f %f %f %f %f\n');
        read_data = reshape(read_data, length(read_data)/num_objects, ...
            num_objects)';
        fclose(fid);
    end
    
    figure(depth_image_fig_num);
    imshow(depth_image, 'border', 'tight');
    depth_image = double(depth_image);
    
    % visualize box info
    for objIdx = 1:num_objects
        
        % box info
        obj_id = read_data(objIdx, 1);
        obj_center_depth = read_data(objIdx, 2);
        obj_x = floor(read_data(objIdx, 3)) + 1;
        obj_y = floor(read_data(objIdx, 4)) + 1;
        obj_s = floor(read_data(objIdx, 5));  % scale
        curColor = COLORS(objIdx, :);
        
        % depth distribution
        depths = zeros(1, obj_s*obj_s);
        depthPos = 0;
        x_min = max(1, obj_x);
        x_max = min(img_W, obj_x+obj_s-1);
        y_min = max(1, obj_y);
        y_max = min(img_H, obj_y+obj_s-1);
        for x = x_min:x_max
            for y = y_min:y_max
                depthPos = depthPos + 1;
                depths(depthPos) = depth_image(y, x);                
            end
        end
        depths(depthPos+1:end) = [];
        figure(objIdx); clf;
        h = histogram(depths, 0:1:255);
        
        % center depth estimation
        center_depths = zeros(1, obj_s*obj_s);
        depthPos = 0;
        center_region_width = center_region_ratio * obj_s;
        center_x_min = max(1, obj_x + floor(0.5*(obj_s-center_region_width)));
        center_x_max = min(img_W, obj_x + floor(0.5*(obj_s+center_region_width)));
        center_y_min = max(1, obj_y + floor(0.5*(obj_s-center_region_width)));
        center_y_max = min(img_H, obj_y + floor(0.5*(obj_s+center_region_width)));
        for x = center_x_min:center_x_max
            for y = center_y_min:center_y_max
                depthPos = depthPos + 1;
                center_depths(depthPos) = depth_image(y, x);                
            end
        end
        center_depths(depthPos+1:end) = [];
        
        % count inliers
        center_depths = sort(center_depths);
        max_num_inlier = 0;
        estimated_center_depth = 0;
        for dIdx = 1:length(center_depths)
            num_inliers = numel( ...
                find(abs(center_depths - center_depths(dIdx))...
                <= 0.5 * depth_foreground_window_size));
            if num_inliers > max_num_inlier
                max_num_inlier = num_inliers;
                estimated_center_depth = center_depths(dIdx);
            end
        end
        
        % depth grouping
        foreground_depths = zeros(1, obj_s*obj_s);
        depthPos = 0;
        for x = x_min:x_max
            for y = y_min:y_max
                cur_depth = depth_image(y, x);
                if abs(cur_depth - estimated_center_depth) > ...
                        0.5 * depth_foreground_window_size
                    continue;
                end
                depthPos = depthPos + 1;
                foreground_depths(depthPos) = cur_depth;                
            end
        end
        foreground_depths(depthPos+1:end) = [];
        
        % find the first modal of foreground depths
        [num_pixels, edges] = histcounts(foreground_depths, min(foreground_depths):max(foreground_depths));
        face_depth = 0;
        for dIdx = 1:length(num_pixels)
            dIdx_min = max(1, max(length(num_pixels)-depth_face_window_size));
            dIdx_max = min(dIdx_min + depth_face_window_size, length(num_pixels));
            if num_pixels(dIdx) == max(num_pixels(dIdx_min:dIdx_max))
                face_depth = edges(dIdx);
                break;
            end
        end
        
        fprintf('[%04d] %d: center depth = %d, face depth = %d\n', ...
            fIdx-1, obj_id, estimated_center_depth, face_depth);
        
        % draw box
        figure(depth_image_fig_num); 
        hold on;
        rectangle('position', [center_x_min, center_y_min, ...
            center_x_max-center_x_min+1, center_y_max-center_y_min+1], ...
            'EdgeColor', [0, 0, 0]);
        rectangle('position', [obj_x, obj_y, obj_s, obj_s], ...
            'EdgeColor', curColor);
        text(obj_x, obj_y, ['\fontsize{10}\color{white}' ...
            int2str(obj_id)],'BackgroundColor', curColor);
        hold off;
    end
    
    pause;
end


%()()
%('')HAANJU.YOO
