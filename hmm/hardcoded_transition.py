class ActionConditionedMLPTransition(nn.Module):
    def __init__(self, state_dim, num_actions, hidden_dim=128):
        super().__init__()
        self.state_dim = state_dim
        self.num_actions = num_actions

        self.wall_idcs = [(0,3),(1,3),(2,3),(3,3),(4,3),(6,3)]
        
        # if action == 0: # down
        #     next_cell[0] += 1
        # elif action == 1: # up
        #     next_cell[0] -= 1
        # elif action == 2: # right
        #     next_cell[1] += 1
        # elif action == 3: # left
        #     next_cell[1] -= 1
        test = torch.zeros(49).to('cuda')
        test[43] = 1
        print(test.reshape((7,7)))

        down = torch.diag_embed(torch.ones(self.state_dim), offset=-7).to('cuda')
        down = down[:-7, :-7]
        down[-7:,-7:] = torch.diag_embed(torch.ones(7)).to('cuda')
        print((down @ test).reshape((7,7)))

        right = torch.diag_embed(torch.ones(self.state_dim), offset=-1).to('cuda')
        right = right[:-1, :-1]
        right[[6,13,20,27,34,41,48], [6,13,20,27,34,41,48]] = 1
        
        for (row, col) in self.wall_idcs:
            idx = col + row * 7
            right[:,idx-1] = 0
            right[idx-1,idx-1] = 1
        print((right @ test).reshape((7,7)))
        
        left = torch.diag_embed(torch.ones(self.state_dim), offset=1).to('cuda')
        left = left[:-1, :-1]
        left[[0,7,14,21,28,35,42], [0,7,14,21,28,35,42]] = 1
        
        for (row, col) in self.wall_idcs:
            idx = col + row * 7
            left[:,idx+1] = 0
            left[idx+1,idx+1] = 1
        print((left @ test).reshape((7,7)))
        
        up = torch.diag_embed(torch.ones(self.state_dim), offset=7).to('cuda')
        up = up[:-7, :-7]
        up[:7,:7] = torch.diag_embed(torch.ones(7)).to('cuda')
        print((up @ test).reshape((7,7)))
        self.actions = [down, up, right, left]
        
    def forward(self, state, action):
        # print(state.reshape((1, 7, 7)))
        # print(action)
        out = torch.stack([self.actions[a] @ state[i] for i, a in enumerate(action)], dim=0)
        out = out / torch.sum(out, dim=-1, keepdim=True)
        # print(out.reshape((1,7,7)))
        # raise ValueError
        return out