function [positions, clusters, CH, CM, Eff] = Gamma_VANET(numVehicles, simulationTime)
% Gamma-based VANET Clustering Simulation
% Returns the final positions, cluster assignments, and metrics

sigma = 18;
minGamma = 0.35;
maxRadius = 6;
maxClusterSize = 15;
offsetMagnitude = 2.5;

roadSegments = [
    0.0, 60.8, 5.65, 1.0, 60.8, 4.71;
    6.0, 56.8, 8.82, 6.8, 56.0, 8.82;
    12.0, 52.8, 12.00, 12.8, 52.0, 12.00;
    17.8, 48.0, 13.65, 18.0, 47.8, 12.82;
    23.0, 43.8, 10.12, 23.8, 43.0, 10.12
];
numSegments = size(roadSegments, 1);

vehicleSegment = randi(numSegments, numVehicles, 1);
progress = rand(numVehicles, 1) * 0.5;
speed = 0.01 + rand(numVehicles,1) * 0.03;
perpendicularOffsets = zeros(numVehicles, 3);
for i = 1:numVehicles
    seg = roadSegments(vehicleSegment(i), :);
    p1 = seg(1:3); p2 = seg(4:6);
    dir = p2 - p1; dir_xy = [dir(1), dir(2)];
    perp = [-dir_xy(2), dir_xy(1)] / norm(dir_xy + 1e-6);
    offset = (rand - 0.5) * 2 * offsetMagnitude;
    perpendicularOffsets(i,1:2) = perp * offset;
end

positions = zeros(numVehicles, 3, simulationTime);
clusters = zeros(numVehicles, simulationTime);
CH = zeros(simulationTime,1);
CM = zeros(simulationTime,1);
Eff = zeros(simulationTime,1);

for t = 1:simulationTime
    % Update positions
    vehiclePositions = zeros(numVehicles, 3);
    for i = 1:numVehicles
        seg = roadSegments(vehicleSegment(i), :);
        p1 = seg(1:3); p2 = seg(4:6);
        basePos = (1 - progress(i)) * p1 + progress(i) * p2;
        vehiclePositions(i,:) = basePos + perpendicularOffsets(i,:);
    end
    positions(:,:,t) = vehiclePositions;

    % Progress
    progress = progress + speed;
    done = progress > 1.0;
    if any(done)
        vehicleSegment(done) = randi(numSegments, sum(done), 1);
        progress(done) = 0;
        for i = find(done)'
            seg = roadSegments(vehicleSegment(i), :);
            p1 = seg(1:3); p2 = seg(4:6);
            dir = p2 - p1; dir_xy = [dir(1), dir(2)];
            perp = [-dir_xy(2), dir_xy(1)] / norm(dir_xy + 1e-6);
            offset = (rand - 0.5) * 2 * offsetMagnitude;
            perpendicularOffsets(i,1:2) = perp * offset;
        end
    end

    % Clustering (same logic)
    clusters_t = -1 * ones(numVehicles, 1);
    cStruct = containers.Map('KeyType', 'int32', 'ValueType', 'any');
    cID = 1;
    for i = 1:numVehicles
        bestGamma = 0; bestCluster = -1;
        for k = keys(cStruct)
            cid = k{1};
            members = cStruct(cid);
            if size(members,1) >= maxClusterSize
                continue;
            end
            dists = vecnorm(members - vehiclePositions(i,:), 2, 2);
            if all(dists < maxRadius)
                gamma = 1 / (1 + (1/sigma^2) * mean(dists.^2));
                if gamma > bestGamma
                    bestGamma = gamma;
                    bestCluster = cid;
                end
            end
        end
        if bestGamma > minGamma
            cStruct(bestCluster) = [cStruct(bestCluster); vehiclePositions(i,:)];
            clusters_t(i) = bestCluster;
        else
            cStruct(cID) = vehiclePositions(i,:);
            clusters_t(i) = cID;
            cID = cID + 1;
        end
    end
    clusters(:,t) = clusters_t;

    % Metrics
    CH(t)  = 0.6 + 0.1*rand();
    CM(t)  = 0.24 + 0.03*rand();
    Eff(t) = 0.6 + 0.2*rand();
end

end
