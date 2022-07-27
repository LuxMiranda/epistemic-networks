import epnets as ep

agents = ep.make_agents(
            n_agents=6, 
            n_credences=5, 
            n_pulls=10
            )

net = ep.EpistemicNetwork(agents, structure='partial_recommender',
        n_recommendations=1, n_partial_links=1, recommend='similar')

print(net)
net.update()
print(net)
net.update()
print(net)
net.update()
print(net)
