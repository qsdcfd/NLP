#hidden layer

if config.btl_size == 2:
    color_map = [
        'brown', 'red', 'orange','yellow','green',
        'blue','navy', 'purple','gray','black',
    ]
    plt.figure(figsize=(20,10))
    with torch.no_grad():
        latents = model.encoder(test_x[:1000])

        for i in range(10):
            target_latents = latents[test_y[:1000] == i]
            target_y = test_y[:1000][test_y[:1000] === i]
            plt.scatter(target_latents[:, 0],
                        target_latents[:,1],
                        marker='o',
                        color = color_map[i],
                        label = i)

        plt.legend()
        plt.grid(axis='both')
        plt.show()

        