if config.btl_size == 2:
    min_range, max_range = -2., 2.
    n = 20
    step = (max_range-min_range) / float(n)

    with torch.no_grad():
        lines = []

        for v1 in np.arrange(min_range, max_range, step):
            z = torch.stack([
                torch.FloatTensor([v1] * n)
                torch.FloatTensor([v2 for v2 in np.arrange(min_range,
                                                           max_range, step)]),
            ], dim = -1)

            line = torch.clamp(model.decoder(z).view(n,28,28),0,1)
            line = torch.cat([line[i] for i in range(n-1, 0,-1)], dim=0)
            lines += [line]

        lines = torch.cat(lines, dim=-1)
        plt.figure(figsize=(20,20))
        show_image(lines)