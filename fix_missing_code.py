# Read the file
with open("exp/kmeans.py", "r") as f:
    lines = f.readlines()

# Find the line with "centroids_future.wait()" and "weights_future.wait()"
insert_after = -1
for i, line in enumerate(lines):
    if "weights_future.wait()" in line:
        insert_after = i
        break

if insert_after == -1:
    print("Could not find insertion point!")
    exit(1)

# The code to insert after the wait() calls
missing_code = """
        # (K)
        weights_total = all_weights.sum(dim=0)

        # make sure that any centroids with zero weights are all equal
        empty_centroids_mask = weights_total == 0
        num_empty_centroids = empty_centroids_mask.sum().item()
        if num_empty_centroids > 0:
            logger.debug(f"Found {num_empty_centroids} centroids with zero weights")

            empty_centroids = all_centroids[:, empty_centroids_mask]
            empty_centroids_rolled = empty_centroids.roll(1, dims=0)
            assert th.allclose(empty_centroids, empty_centroids_rolled), (
                f"Empty centroids are not equal: {empty_centroids} != {empty_centroids_rolled}"
            )

        # (N, K) - handle division by zero for empty centroids
        weights_proportion = th.where(
            weights_total.unsqueeze(0) > 0,
            all_weights / weights_total.unsqueeze(0),
            th.full_like(all_weights, fill_value=1 / world_size),
        )

        if th.isnan(weights_proportion).any():
            logger.error(
                f"NaN values found in weights_proportion! weights_total: {weights_total}"
            )
            logger.error(f"all_weights: {all_weights}")

        # (K, D)
        new_centroids = (all_centroids * weights_proportion.unsqueeze(-1)).sum(dim=0)

"""

# Insert the code
lines.insert(insert_after + 1, missing_code)

# Write back
with open("exp/kmeans.py", "w") as f:
    f.writelines(lines)

print("Missing code inserted successfully!")
