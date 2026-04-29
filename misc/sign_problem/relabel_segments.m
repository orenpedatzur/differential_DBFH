function seg_id = relabel_segments(seg_id)
    labels = unique(seg_id(:));
    labels(labels == 0) = [];

    if isempty(labels)
        seg_id = zeros(size(seg_id), 'uint32');
        return;
    end

    max_label = max(double(labels(:)));
    new_id = zeros(max(max_label, 1), 1, 'uint32');

    for k = 1:numel(labels)
        new_id(double(labels(k))) = uint32(k);
    end

    mask = seg_id > 0;
    seg_id = uint32(seg_id);
    seg_id(mask) = new_id(double(seg_id(mask)));
end