figure
cell_keypoint = table2cell(tbl_keypoints);
for i=1:100
    img = imread(cell2mat(cell_keypoint(i,1)));    
    imshow(img); hold on

    for j=2:18
        a=(cell2mat(keypoints(i,j)));
        scatter(a(:,1),a(:,2));
        hold on
    end
    drawnow
    hold off
    pause(1)
end

for i=1:100
    coco_kpt(i).annorect.keypoints
end