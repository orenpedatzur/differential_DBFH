function [edge_i, edge_j] = build_segment_edges(seg_id)
% Build nearest-neighbor equality edges inside each nonzero segment.

    [Ny, Nx] = size(seg_id);
    lin = reshape(1:Ny*Nx, Ny, Nx);

    % horizontal neighbors
    same_h = seg_id(:,1:end-1) > 0 & seg_id(:,1:end-1) == seg_id(:,2:end);
    i_h = lin(:,1:end-1);
    j_h = lin(:,2:end);

    % vertical neighbors
    same_v = seg_id(1:end-1,:) > 0 & seg_id(1:end-1,:) == seg_id(2:end,:);
    i_v = lin(1:end-1,:);
    j_v = lin(2:end,:);

    edge_i = [i_h(same_h); i_v(same_v)];
    edge_j = [j_h(same_h); j_v(same_v)];
end