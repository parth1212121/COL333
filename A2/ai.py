import time
import math
import random
import numpy as np              
from helper import *


class AIPlayer:

    

    def __init__(self, player_number: int, timer):
        """
        Intitialize the AIPlayer Agent

        # Parameters
        `player_number (int)`: Current player number, num==1 starts the game
        
        `timer: Timer`
            - a Timer object that can be used to fetch the remaining time for any player
            - Run `fetch_remaining_time(timer, player_number)` to fetch remaining time of a player
        """
        self.player_number = player_number
        self.type = 'ai'
        self.player_string = 'Player {}: ai'.format(player_number)
        self.timer = timer
        self.possible_actions = []
        self.state  = []
        self.union_connections = []
        self.last_move = (-5,-5)
        self.my_last_move = (0,0)
        self.game_setup_initialised = False
        self.visit = []
        self.wins = []
        self.corner_cells = []
        self.edge_cells = []
        self.game_size = 4
        self.time_limit = 5
        self.winning_edge_weight = 3
        self.wining_vertex_weight = 6
        self.priority_weight = []
        self.decremental_value = 0.01
        self.start_time = time.time()
        


    def reset_union_connections(self):
        n = self.game_size
        m = self.game_size
        self.union_connections = np.zeros((n*m,4),dtype = int)
        for i in range(n):
            for j in range(m):
                self.visit[i][j] = 1
                self.union_connections[i*len(self.state)+j][0] = -1
                self.union_connections[i*len(self.state)+j][1] = 0
                self.union_connections[i*len(self.state)+j][2] = 0
                self.union_connections[i*len(self.state)+j][3] = 0
        power = [ 1,2,4,8,16,32,62,128]
        for i in range(6):
            x = int(i)
            t = self.get_vertices_on_edge(x)
            tt= self.get_vertex_at_corner(x)
            #print(self.game_size,t,tt)
            for cell in t:
                self.union_connections[cell[0]*len(self.state)+cell[1] ][2] += power[i]
            self.union_connections[tt[0]*len(self.state)+tt[1] ][3] += power[i]
        for move in self.move_history:
            self.union_find(move[0],move[1],move[2])
        return
    def get_vertices_on_edge(self,edge):
        z = []
        if edge == 0:
            for i in range(self.game_size,self.game_size*2-2):
                z.append((0,i))

        if edge == 1:
            for i in range(1,self.game_size-1):
                z.append((i,self.game_size*2-2))
        if edge == 2:
            d = 1
            for i in range(self.game_size,self.game_size*2-2):
                z.append((i,self.game_size*2-2 - d))
                d+=1
        if edge == 3:
            for i in range(1,self.game_size-1):
                z.append((i+self.game_size-1,i))
        if edge == 4:
            for i in range(1,self.game_size-1):
                z.append((i,0))
        if edge == 5:
            for i in range(1,self.game_size-1):
                z.append((0,i))
        return z
    def get_vertex_at_corner(self,corner):
        if corner == 0:
            return (0,self.game_size-1)
        if corner == 1:
            return (0,self.game_size*2 - 2)
        if corner == 2:
            return (self.game_size-1,self.game_size*2-2)
        if corner == 3:
            return (self.game_size*2-2,self.game_size - 1)
        if corner == 4:
            return (self.game_size-1,0)
        if corner == 5:
            return (0,0)
    def Initialise_game(self,state:np.array,player_number:int):
        n = len(state)
        m = len(state[1])
        self.state = np.zeros((n,m))
        self.visit = np.zeros((n,m))
        self.wins = np.zeros((n,m))
        self.game_size = int ( (n + 1) / 2)

        self.union_connections = np.zeros((n*m,4),dtype = int)
        self.priority_weight = np.zeros((n,m))
        for i in range(n):
            for j in range(m):
                self.visit[i][j] = 1
                self.union_connections[i*len(self.state)+j][0] = -1
                self.union_connections[i*len(self.state)+j][1] = 0
                self.union_connections[i*len(self.state)+j][2] = 0
                self.union_connections[i*len(self.state)+j][3] = 0
                
                if(state[i][j]==3):
                    self.state[i][j] = 3
                elif(state[i][j]==3-player_number):
                    self.last_move = (i,j)  
                    self.state[i][j] = 3-player_number
                    self.union_find(i,j,3-player_number)
                else:
                    self.possible_actions.append((i,j))

        corner_cells = []
        edge_cells = []
        self.edge_cells = []
        power = [ 1,2,4,8,16,32,62,128]
        for i in range(6):
            x = int(i)
        
            t = self.get_vertices_on_edge(x)
            tt= self.get_vertex_at_corner(x)
            #print(self.game_size,t,tt)
            for cell in t:
                edge_cells.append(cell)
                self.union_connections[cell[0]*len(self.state)+cell[1] ][2] += power[i]
            corner_cells+=[tt]
            self.union_connections[tt[0]*len(self.state)+tt[1] ][3] += power[i]
        
        
        self.edge_cells = np.array(edge_cells)
        #self.possible_actions = np.array(self.possible_actions)
        self.corner_cells = corner_cells
        self.union_connections = np.array(self.union_connections) 
    def union_find(self,i,j,player_number):

        neighbor = self.get_immediate_connections(i,j)
        world = len(self.state)
        #print(world,"length of state")
        self.union_connections[i*world+j][0] = i*world+j
        self.union_connections[i*world+j][1] = player_number
        notFound = True
        for n in neighbor:
            
            if(self.state[n[0]][n[1]]==player_number and notFound):
                
                update_union_posn = [n[0]*world+n[1]]
                x = n[0]
                y = n[1]
                #print(x,y,end="->")
                while( self.union_connections[x*world+y][0]!=(x*world+y) and self.union_connections[x*world+y][1]==player_number) :
                    update_union_posn+=[x*world+y]
                    x1 = x
                    y1 = y
                    x = self.union_connections[x1*world+y1][0]//world
                    y = self.union_connections[x1*world+y1][0] % world
                    #print(x,y,end="->")
                #print("T")
                self.union_connections[x*world+y][2] = ( self.union_connections[x*world+y][2] | self.union_connections[i*world+j][2] )
                self.union_connections[x*world+y][3] = ( self.union_connections[x*world+y][3] | self.union_connections[i*world+j][3] )

                for posn in update_union_posn:
                    self.union_connections[posn][0] = x*world+y
            
                self.union_connections[i*world+j][0] = x*world+y
                

                notFound = False
            elif (self.state[n[0]][n[1]] == player_number):
                update_union_posn = [n[0]*world+n[1]]
                x = n[0]
                y = n[1]
                #print(x,y,self.union_connections[x*world+y][0],end="->")
                
                while( self.union_connections[x*world+y][0]!=(x*world+y) and self.union_connections[x*world+y][1]==player_number) :
                    update_union_posn+=[x*world+y]
                    x1 = x
                    y1 = y
                    x = self.union_connections[x1*world+y1][0]//world
                    y = self.union_connections[x1*world+y1][0] % world
                    #print(x,y,self.union_connections[x*world+y][0],end="->")

                #print("T")
                for posn in update_union_posn:
                    self.union_connections[posn][0] = self.union_connections[i*world+j][0] 
                    

                self.union_connections[x*world+y][0] = self.union_connections[i*world+j][0] 
                self.union_connections[self.union_connections[i*world+j][0] ][2] = ( self.union_connections[self.union_connections[i*world+j][0] ][2]  |  self.union_connections[x*world+y][2])
                self.union_connections[self.union_connections[i*world+j][0] ][3] = ( self.union_connections[self.union_connections[i*world+j][0] ][3]  |  self.union_connections[x*world+y][3])

        return
    def get_connected_edge(self,action):
        x = action[0]
        y = action[1]
        world = len(self.state)
        while( self.union_connections[x*world+y][0]!=(x*world+y) ) :
            x1 = x
            y1 = y
            x = self.union_connections[x1*world+y1][0]//world
            y = self.union_connections[x1*world+y1][0] % world
        

        return self.union_connections[x*world+y][2]
    def get_connected_vertex(self,action):
        
        x = action[0]
        y = action[1]
        world = len(self.state)
        while( self.union_connections[x*world+y][0]!=(x*world+y) ) :
           
            x1 = x
            y1 = y
            x = self.union_connections[x1*world+y1][0]//world
            y = self.union_connections[x1*world+y1][0] % world
        return self.union_connections[x*world+y][3]
    def check_virtual_fork(self,player_number,action):
        
        q = []
        i = 0
        q+=[action]
        world = len(self.state)
        dd = self.union_connections[action[0]*world+action[1]][2]
        seen = dict()
        vertex = set()
        while(i<len(q)):
            
            if( (q[i][0]*world + q[i][1]) not in seen):
                seen[q[i][0] * world + q[i][1]] = 1
                if(q[i][0] != action[0] or q[i][1] != action[1]):
                    dd = ( self.get_connected_edge(q[i]) | dd )
                gg = dd
                for ppp in range(6):
                    vertex.add((gg%2)*(ppp+1))
                    gg = gg//2
                
                if 0 in vertex:
                    vertex.remove(0)
                if(len(vertex) > 2):
                    return True

                v_neighbor = self.get_virtual_connections(q[i][0],q[i][1])
                normal_neighbor = self.get_immediate_connections(q[i][0],q[i][1])
                neighbor = []

                for ll in normal_neighbor:
                    neighbor+=[ll]
                for ll in v_neighbor:
                    neighbor+=[(ll[0][0],ll[0][1])]
                
                for r in neighbor :
                    if (self.state[r[0]][r[1]] == player_number):
                        q+=[r]
                        
                        
            i+=1
        return False
    def check_virtual_bridge(self,player_number,action):

        q = []
        i = 0
        q+=[action]
        world = len(self.state)
        dd = self.union_connections[action[0]*world+action[1]][3]
        seen = dict()
        vertex = set()
        while(i<len(q)):
            if( (q[i][0]*world + q[i][1]) not in seen):
                seen[q[i][0]*world + q[i][1]] = 1
                if(q[i][0] != action[0] or q[i][1] != action[1]):
                    dd = ( self.get_connected_vertex(q[i]) | dd )
                gg = dd
                for ppp in range(6):
                    vertex.add((gg%2)*(ppp+1))
                    gg = gg//2
                if 0 in vertex:
                    vertex.remove(0)    
                if(len(vertex) > 1):
                    return True
              
                v_neighbor = self.get_virtual_connections(q[i][0],q[i][1])
                normal_neighbor = self.get_immediate_connections(q[i][0],q[i][1])
                neighbor = []

                for ll in normal_neighbor:
                    neighbor+=[ll]
                for ll in v_neighbor:
                    neighbor+=[(ll[0][0],ll[0][1])]
                
                for r in neighbor :
                
                    if (self.state[r[0]][r[1]] == player_number):
                        q+=[r]
                        
                        
                        
            i+=1
        return False
    def check_ring(self, player_number: int, action: Tuple[int, int]) -> bool:
        # Add the action to the board temporarily
        self.state[action[0]][action[1]] = player_number
        dim = self.state.shape[0]
        siz = dim // 2
        init_move = action
        directions = ["up", "top-left", "bottom-left", "down"]
        visited = set()
        neighbors = self.get_immediate_connections(action[0], action[1])
        neighbors = [self.state[neighbor[0]][neighbor[1]] == player_number for neighbor in neighbors]

        # Trivially false if less than 2 True neighbors present
        if neighbors.count(True) < 2:
            self.state[action[0]][action[1]] = 0
            return False





        # In the first step, move in 4 contiguous directions to check for a ring
        exploration = []
        for direction in directions:
            x, y = action
            half = np.sign(action[1] - siz)  # 0 for mid, -1 for left, 1 for right
            #print("debug --> ")
            #print(direction)
            #print(half)
            #print("debug close--> ")
            direction_coors = move_coordinates(direction, half)
            nx, ny = x + direction_coors[0], y + direction_coors[1]
            if 0 <= nx < dim and 0 <= ny < dim and self.state[nx, ny] == player_number:
                exploration.append(((nx, ny), direction))
                visited.add((nx, ny, direction))

        ring_length = 1
        # In the later steps, move in 3 "forward" directions (avoids sharp turns)


        
        while exploration:
            new_exp = []
            for to_explore in exploration:
                move, prev_direction = to_explore
                x, y = move
                half = np.sign(y - siz)
                new_directions = three_forward_moves(prev_direction)
                for direction in new_directions:
                    direction_coors = move_coordinates(direction, half)
                    nx, ny = x + direction_coors[0], y + direction_coors[1]
                    if is_valid(nx, ny, dim) and self.state[nx, ny] == player_number and (nx, ny, direction) not in visited:
                        if init_move == (nx, ny) and ring_length >= 5:
                            # Restore the action and return True if a ring is found
                            self.state[action[0]][action[1]] = 0
                            return True
                        new_exp.append(((nx, ny), direction))
                        visited.add((nx, ny, direction))
            exploration = new_exp
            ring_length += 1

        # Restore the action and return False if no ring found
        self.state[action[0]][action[1]] = 0
        return False
    def check_fork(self,player_number,action):

        neighbor = self.get_immediate_connections(action[0],action[1])
        edges = set()
        for n in neighbor :
            if (self.union_connections[n[0]*len(self.state)+n[1]][1] == player_number):
                dd = self.get_connected_edge(n)
                dd = ( self.union_connections[action[0]*len(self.state)+action[1]][2] | dd )
                for i in range(6):
                    edges.add((dd%2)*(i+1))
                    dd = dd//2
                

            if 0 in edges:
                edges.remove(0)
            #self.priority_weight[action[0]][action[1]] = len(edges)*self.winning_edge_weight
            if(len(edges) > 2):
                return True

        return False
    def check_bridge(self,player_number,action):

        neighbor = self.get_immediate_connections(action[0],action[1])
        vertex = set()
        for n in neighbor :
            if (self.union_connections[n[0]*len(self.state)+n[1]][1] == player_number):
                dd = self.get_connected_vertex(n)
                dd = ( self.union_connections[action[0]*len(self.state)+action[1]][3] | dd )

                for i in range(6):
                    vertex.add((dd%2)*(i+1))
                    dd = dd//2

            if 0 in vertex:
                vertex.remove(0)
            #self.priority_weight[action[0]][action[1]] = len(vertex)*self.wining_vertex_weight
            if(len(vertex) > 1):
                return True
        return False
    def check_win_moves(self,player_number):
        win_move = []
        for i in range(len(self.state)):
            for j in range(len(self.state[1])):
                if(self.state[i][j] == 0):

                    result1 = self.check_bridge(player_number,(i,j))
                    result2 = self.check_fork(player_number,(i,j))
                    result3 = self.check_ring(player_number,(i,j))
                    


                    if( result1 or result2 or result3 ):
                        # print("possible win_moves ---->")
                        # print(i, "  ----   ",j)
                        win_move.append((i,j))
        


        return win_move  
    def check_win_moves_virtual(self,player_number):
        win_move =[]
        for i in range(len(self.state)):
            for j in range(len(self.state[0])):
                if(self.state[i][j]==0):
                    result4 = self.check_virtual_bridge(player_number,(i,j))
                    result5 = self.check_virtual_fork(player_number,(i,j))
                    if(result4 or result5):
                        # print("possible virtual win move --->")
                        # print(i, "  ----   ",j)

                        win_move.append((i,j))


        return win_move  
    
    
    def get_closest_next_moves(self,player_number):
        
        evaluated_moves = []
        # apply the hueristics on these closest_move and filter out the best 20...
        n = len(self.state)
        m = len(self.state[0])

        favourable_moves_me = np.zeros((n,m))
        favourable_moves_opp = np.zeros((n,m))
        visited_me = np.zeros((n,m))
        visited_opp = np.zeros((n,m))

        for i in range(n):
            for j in range(m):
                favourable_moves_me[i][j] = 10000
                favourable_moves_opp[i][j] = 10000

        stack_me = []
        stack_opp = []

        my_vertices = []
        opp_vertices = []
        my_edges = []
        opp_edges = []
        my_vertex_value = self.get_connected_vertex(self.my_last_move)
        opp_vertex_value = self.get_connected_vertex(self.last_move)
        my_edge_value = self.get_connected_edge(self.my_last_move)
        opp_edge_value = self.get_connected_edge(self.last_move)

        for i in range(6):
            if(my_vertex_value%2==1):
                my_vertices.append(i)
            if(opp_vertex_value%2==1):
                opp_vertices.append(i)
            if(my_edge_value%2==1):
                my_edges.append(i)
            if(opp_edge_value%2==1):
                opp_edges.append(i)

            my_vertex_value = my_vertex_value//2
            opp_vertex_value = opp_vertex_value//2
            my_edge_value = my_edge_value//2
            opp_edge_value = opp_edge_value//2
        
        


        for ws in my_vertices:
            i  = self.get_vertex_at_corner(ws)
            if(self.state[i[0]][i[1]] == player_number):
                stack_me.append((i,0))
                favourable_moves_me[i[0]][i[1]] = 0
            elif(self.state[i[0]][i[1]]== 0 ):
                favourable_moves_me[i[0]][i[1]] = 1
                stack_me.append((i,1))
        
        for ws in opp_vertices:
            i = self.get_vertex_at_corner(ws)
            if(self.state[i[0]][i[1]] == 3 - player_number):
                stack_opp.append((i,0))
                favourable_moves_opp[i[0]][i[1]] = 0
            elif(self.state[i[0]][i[1]]== 0 ):
                favourable_moves_opp[i[0]][i[1]] = 1
                stack_opp.append((i,1))
   

        for ws in my_edges:
            for i in self.get_vertices_on_edge(ws):
                if(self.state[i[0]][i[1]] == player_number):
                    stack_me.append((i,0))
                    favourable_moves_me[i[0]][i[1]] = 0
                elif(self.state[i[0]][i[1]]== 0 ):
                    favourable_moves_me[i[0]][i[1]] = 1
                    stack_me.append((i,1))
        
        for ws in opp_edges:
            for i in self.get_vertices_on_edge(ws):
                if(self.state[i[0]][i[1]] == 3 - player_number):
                    stack_opp.append((i,0))
                    favourable_moves_opp[i[0]][i[1]] = 0
                elif(self.state[i[0]][i[1]]== 0 ):
                    favourable_moves_opp[i[0]][i[1]] = 1
                    stack_opp.append((i,1))

        while(len(stack_me)>0):
            stack_me.sort(key = lambda x: x[1] )
            i = stack_me[0][0][0]
            j = stack_me[0][0][1]
            d = stack_me[0][1]
            
            if(i==self.my_last_move[0] and j==self.my_last_move[1]):
                break
            stack_me.pop(0)
            if(visited_me[i][j] == 0):
                
                visited_me[i][j] = 1
                neighbor = self.get_immediate_connections(i,j)
                vn = self.get_virtual_connections(i,j)

                for N in neighbor:
                    if( self.state[N[0]][N[1]] == 0):
                        if(favourable_moves_me[N[0]][N[1]] > favourable_moves_me[i][j] + 1):
                            favourable_moves_me[N[0]][N[1]] = favourable_moves_me[i][j] + 1
                            stack_me.append((N,favourable_moves_me[i][j] + 1))
                    elif (self.state[N[0]][N[1]] == player_number):
                        if(favourable_moves_me[N[0]][N[1]] > favourable_moves_me[i][j] ):
                            favourable_moves_me[N[0]][N[1]] = favourable_moves_me[i][j]
                            stack_me.append((N,favourable_moves_me[i][j]))
        
                for NNN in vn:
                    N = NNN[0]
                    if( self.state[N[0]][N[1]] == 0):
                        if(favourable_moves_me[N[0]][N[1]] > favourable_moves_me[i][j] + 1):
                            favourable_moves_me[N[0]][N[1]] = favourable_moves_me[i][j] + 1
                            stack_me.append((N,favourable_moves_me[i][j] + 1))
                    elif (self.state[N[0]][N[1]] == player_number):
                        if(favourable_moves_me[N[0]][N[1]] > favourable_moves_me[i][j] ):
                            favourable_moves_me[N[0]][N[1]] = favourable_moves_me[i][j]
                            stack_me.append((N,favourable_moves_me[i][j]))

        while(len(stack_opp)>0):
            stack_opp.sort(key = lambda x: x[1] )
            i = stack_opp[0][0][0]
            j = stack_opp[0][0][1]
            d = stack_opp[0][1]
            
            if(i==self.my_last_move[0] and j==self.my_last_move[1]):
                break
            stack_opp.pop(0)
            if(visited_opp[i][j] == 0):
                
                visited_opp[i][j] = 1
                neighbor = self.get_immediate_connections(i,j)
                vn = self.get_virtual_connections(i,j)

                for N in neighbor:
                    if( self.state[N[0]][N[1]] == 0):
                        if(favourable_moves_opp[N[0]][N[1]] > favourable_moves_opp[i][j] + 1):
                            favourable_moves_opp[N[0]][N[1]] = favourable_moves_opp[i][j] + 1
                            stack_opp.append((N,favourable_moves_opp[i][j] + 1))
                    elif (self.state[N[0]][N[1]] == 3 - player_number):
                        if(favourable_moves_opp[N[0]][N[1]] > favourable_moves_opp[i][j] ):
                            favourable_moves_opp[N[0]][N[1]] = favourable_moves_opp[i][j]
                            stack_opp.append((N,favourable_moves_opp[i][j]))
        
                for NNN in vn:
                    N = NNN[0]
                    if( self.state[N[0]][N[1]] == 0):
                        if(favourable_moves_opp[N[0]][N[1]] > favourable_moves_opp[i][j] + 1):
                            favourable_moves_opp[N[0]][N[1]] = favourable_moves_opp[i][j] + 1
                            stack_opp.append((N,favourable_moves_opp[i][j] + 1))
                    elif (self.state[N[0]][N[1]] == 3 - player_number):
                        if(favourable_moves_opp[N[0]][N[1]] > favourable_moves_opp[i][j] ):
                            favourable_moves_opp[N[0]][N[1]] = favourable_moves_opp[i][j]
                            stack_opp.append((N,favourable_moves_opp[i][j]))


        
        

        for move in self.possible_actions:
            rr = self.evaluate_state(move,player_number)
            gg = rr
            if(player_number==self.player_number):
                
                if move in self.get_immediate_connections(self.my_last_move[0],self.my_last_move[1]):
                    rr=gg+10*max(0,4 - favourable_moves_me[move[0]][move[1]])
                for rrr in self.get_virtual_connections(self.my_last_move[0],self.my_last_move[1]):
                    kkk = rrr[0]
                    if kkk[0] == move[0] and kkk[1] == move[1]:
                        rr+=10*max(0,4 - favourable_moves_me[move[0]][move[1]])

                if move in self.get_immediate_connections(self.last_move[0],self.last_move[1]):
                    rr+=10*max(0,4 - favourable_moves_opp[move[0]][move[1]])
                for rrr in self.get_virtual_connections(self.last_move[0],self.last_move[1]):
                    kkk = rrr[0]
                    if kkk[0] == move[0] and kkk[1] == move[1]:
                        rr+=10*max(0,4 - favourable_moves_opp[move[0]][move[1]]) 
            else:
                if move in self.get_immediate_connections(self.my_last_move[0],self.my_last_move[1]):
                    rr=gg+10*max(0,4 - favourable_moves_opp[move[0]][move[1]])
                for rrr in self.get_virtual_connections(self.my_last_move[0],self.my_last_move[1]):
                    kkk = rrr[0]
                    if kkk[0] == move[0] and kkk[1] == move[1]:
                        rr+=10*max(0,4 - favourable_moves_opp[move[0]][move[1]])

                if move in self.get_immediate_connections(self.last_move[0],self.last_move[1]):
                    rr+=10*max(0,4 - favourable_moves_me[move[0]][move[1]])
                for rrr in self.get_virtual_connections(self.last_move[0],self.last_move[1]):
                    kkk = rrr[0]
                    if kkk[0] == move[0] and kkk[1] == move[1]:
                        rr+=10*max(0,4 - favourable_moves_me[move[0]][move[1]])     
            evaluated_moves.append((move,rr))   
            

        
        
            

        # Sort the evaluated moves by score in descending order
        evaluated_moves.sort(key=lambda x: x[1], reverse=True)

        # Select the top 20 moves (or fewer if there are not enough moves)
        highest_score = evaluated_moves[0][1]   
        top_moves = []
        for move, score in evaluated_moves:
            if score >= highest_score*0.7:
                top_moves.append((move,score))
            else:
                break

        closest_next_move=top_moves


        return closest_next_move
    def get_no_of_connected_comp(self,player_number):
        w = []
        for i in range(len(self.state)):
            for j in range(len(self.state)):
                if(self.union_connections[ i*len(self.state) + j][0] == i*len(self.state) + j ):
                    w+= [self.union_connections[i*len(self.state) + j]]
        ans =0
        for i in w:
            if (i[1]==player_number):
                ans+=1

        return ans
            
        #w = [[ #ID of Connected component , Player number , 6 bit number denoting edges , 6 bit number denoting vertices]]
    def evaluate_state(self, action, player_number): 
        
        score = 0
        
        if(player_number == self.player_number):
            vn1 = self.get_virtual_connections(self.my_last_move[0],self.my_last_move[1])
            for i in vn1:
                if(self.state[i[0][0]][i[0][1]]==player_number):
                    if(action[0] == i[1][0] and action[1] == i[1][1] and self.state[i[2][0]][i[2][1]]==3 - player_number):
                        score +=30
                        break
                    if(action[0] == i[2][0] and action[1] == i[2][1] and self.state[i[1][0]][i[1][1]]==3 - player_number):
                        score +=30
                        break
        if(player_number == 3 - self.player_number):
            vn1 = self.get_virtual_connections(self.last_move[0],self.last_move[1])
            for i in vn1:
                if(self.state[i[0][0]][i[0][1]]==player_number):
                    if(action[0] == i[1][0] and action[1] == i[1][1] and self.state[i[2][0]][i[2][1]]==3 - player_number):
                        score +=30
                        break
                    if(action[0] == i[2][0] and action[1] == i[2][1] and self.state[i[1][0]][i[1][1]]==3 - player_number):
                        score +=30
                        break
        vn1 = self.get_virtual_connections(action[0],action[1])
        
        for i in vn1 :
            
            if(self.state[i[0][0]][i[0][1]]==player_number):
                score+=20
                break
        for i in vn1 :
            if(self.state[i[0][0]][i[0][1]]==3 - player_number):
                score+=10
                break

        
        
        reduced = set()
        reduced2 = set()

        
        gg = self.get_immediate_connections(action[0],action[1])


        world = len(self.state)
        for pp in gg:
            if(self.state[pp[0]][pp[1]]==player_number):
                x = pp[0]
                y = pp[1]
                #print(x,y,end="->")
                while( self.union_connections[x*world+y][0]!=(x*world+y) and self.union_connections[x*world+y][1]==player_number) :
                    x1 = x
                    y1 = y
                    x = self.union_connections[x1*world+y1][0]//world
                    y = self.union_connections[x1*world+y1][0] % world
                reduced.add(x*world + y)
            if(self.state[pp[0]][pp[1]]==3-player_number):
                x = pp[0]
                y = pp[1]
                #print(x,y,end="->")
                while( self.union_connections[x*world+y][0]!=(x*world+y) and self.union_connections[x*world+y][1]==3-player_number) :
                    x1 = x
                    y1 = y
                    x = self.union_connections[x1*world+y1][0]//world
                    y = self.union_connections[x1*world+y1][0] % world
                reduced2.add(x*world + y)

        if len(reduced)>1 :
            score += 30
        if len(reduced2)>1 :
            score += 30

        return score
    def best_next_move_for_me(self):


        # print(self.state)

        kawasaki = self.check_win_moves(self.player_number)
        if(len(kawasaki)>0):
            
            # print("kawasaki")
            # print(kawasaki)

            #print("kawasaki :)",kawasaki)
            return kawasaki[0],True
        
        cargo = self.check_win_moves(3-self.player_number)
        if(len(cargo)>0):

            # print("cargo--->")
            # print(cargo)
            
        #print("cargo block:( ",cargo)
            return cargo[0],False
        
        virtual_kawasaki = self.check_win_moves_virtual(self.player_number)
        if ( len(virtual_kawasaki) > 0 ):
            # print("Virtual kawasaki")
            # print(virtual_kawasaki)
            return virtual_kawasaki[0],True
        virtual_cargo = self.check_win_moves_virtual(3-self.player_number)
        if ( len(virtual_cargo)>0):
            # print("virtual cargo")
            # print(virtual_cargo)
            return virtual_cargo[0],False
        # print(" No winning move found for opp")
        # print("Doing Heuristic move")
        return (-1,-1),False
        
        

        return random.choice(self.possible_actions),False
        return max(self.possible_actions, key = lambda x: ( self.wins[x[0]][x[1]] / self.visit[x[0]][x[1]] +self.priority_weight[x[0]][x[1]]+  (exploration_weights * np.sqrt( np.log(self.visit[action[0]][action[1]]) / self.visit[x[0]][x[1]])  )) ),False
    def best_next_move_for_opponent(self):

    
        kawasaki = self.check_win_moves(3 - self.player_number)
        if(len(kawasaki)>0):
            
            # print("kawasaki")
            # print(kawasaki)

            #print("kawasaki :)",kawasaki)
            return kawasaki[0],True
        
        cargo = self.check_win_moves(self.player_number)
        if(len(cargo)>0):

            # print("cargo--->")
            # print(cargo)
            
        #print("cargo block:( ",cargo)
            return cargo[0],False
        
        virtual_kawasaki = self.check_win_moves_virtual(3 - self.player_number)
        if ( len(virtual_kawasaki) > 0 ):
            # print("Virtual kawasaki")
            # print(virtual_kawasaki)
            return virtual_kawasaki[0],True
        virtual_cargo = self.check_win_moves_virtual(self.player_number)
        if ( len(virtual_cargo)>0):
            # print("virtual cargo")
            # print(virtual_cargo)
            return virtual_cargo[0],False
        #print("Random move")
        return (-1,-1),False
        return random.choice(self.possible_actions),False
        
        return max(self.possible_actions, key = lambda x: ( self.wins[x[0]][x[1]] / self.visit[x[0]][x[1]] +self.priority_weight[x[0]][x[1]]+  (exploration_weights * np.sqrt( np.log(self.visit[action[0]][action[1]]) / self.visit[x[0]][x[1]])  )) ),False
    def get_opponent_move(self,state:np.array):
        n = len(state)
        m = len(state[1])
        for i in range(n):
            for j in range(m):
                if(state[i][j]!=self.state[i][j] ):
                    self.state[i][j] = state[i][j]
                    self.last_move = (i,j)
                    if (i,j) in self.possible_actions:
                        self.possible_actions.remove((i,j))
                    self.union_find(i,j,3-self.player_number)
                    #print("opponent move",self.last_move)  

                    break
    def get_neighbor_in_dir(self,i,j,dir):
        world = self.game_size
        if(dir == 0 ):
            return (i-1,j)
        if(dir == 1 ):
            if ( j < world - 1):
                return (i,j+1)
            else:
                return (i-1,j+1)
        if(dir == 2 ):
            if ( j < world - 1):
                return (i+1,j+1)
            else:
                return (i,j+1)
        if(dir == 3 ):
            return (i+1,j)
        if(dir == 4 ):
            if ( j < world  ):
                return (i,j-1)
            else:
                return (i+1,j-1)
        if(dir == 5 ):
            if ( j < world ):
                return (i-1,j-1)
            else:
                return (i,j-1)      
    def get_virtual_connections(self, i , j ):

        virtual_neighbors = []
        for K in range(6):
            g = (-1,-1)
            f = self.get_neighbor_in_dir(i,j,K)

            if(f[0]>-1 and f[1]>-1 and f[0]<len(self.state) and f[1]<len(self.state[0])):
                if(self.state[f[0]][f[1]]==0):
                    g = self.get_neighbor_in_dir(f[0],f[1],(K+1)%6)
            if(g[0]>-1 and g[1]>-1 and g[0]<len(self.state) and g[1]<len(self.state[0])):
                if(self.state[g[0]][g[1]]!=3 ):
                    h = self.get_neighbor_in_dir(i,j,(K+1)%6)
                    if(self.state[h[0]][h[1]]==0):
                        virtual = []
                        virtual.append(g)
                        virtual.append(f)
                        virtual.append(h)
                        virtual_neighbors.append(virtual)

                
            
        return virtual_neighbors
    def get_immediate_connections(self, i , j ):
        
        immediate_neighbors = []
        if ( j < self.game_size -1  ):
            if ( i - 1 >= 0 and j < len(self.state[0]) ):
                if(self.state[i-1][j]!=3):
                    immediate_neighbors.append((i-1,j))
            if ( i < len(self.state) and j + 1 < len(self.state[0]) ):
                if(self.state[i][j+1]!=3):
                    immediate_neighbors.append((i,j+1))
            if ( i +1 < len(self.state) and j + 1 < len(self.state[0]) ):
                if(self.state[i+1][j+1]!=3):
                    immediate_neighbors.append((i+1,j+1))
            if ( i + 1 < len(self.state) and j < len(self.state[0]) ):
                if(self.state[i+1][j]!=3):
                    immediate_neighbors.append((i+1,j))
            if ( i < len(self.state) and j - 1 >= 0 ):
                if(self.state[i][j-1]!=3):
                    immediate_neighbors.append((i,j-1))
            if ( i - 1 >= 0 and j - 1 >= 0 ):
                if(self.state[i-1][j-1]!=3):
                    immediate_neighbors.append((i-1,j-1))
            

        elif ( j > self.game_size -1 ):
            if ( i - 1 >= 0 and j < len(self.state[0]) ):
                if(self.state[i-1][j]!=3):
                    immediate_neighbors.append((i-1,j))
            if ( i - 1 >= 0 and j + 1  < len(self.state[0]) ):
                if(self.state[i-1][j+1]!=3):
                    immediate_neighbors.append((i-1,j+1))
            if ( i < len(self.state) and j + 1 < len(self.state[0]) ):
                if(self.state[i][j+1]!=3):
                    immediate_neighbors.append((i,j+1))
            if ( i + 1 < len(self.state) and j < len(self.state[0]) ):
                if(self.state[i+1][j]!=3):
                    immediate_neighbors.append((i+1,j))
            if ( i +1 < len(self.state) and j - 1 >=0 ):
                if(self.state[i+1][j-1]!=3):
                    immediate_neighbors.append((i+1,j-1))
            if ( i < len(self.state) and j - 1 >= 0 ):
                if(self.state[i][j-1]!=3):
                    immediate_neighbors.append((i,j-1))

        else:
            if ( i - 1 >= 0 and j < len(self.state[0]) ):
                if(self.state[i-1][j]!=3):
                    immediate_neighbors.append((i-1,j))
            if ( i - 1 >= 0 and j + 1 < len(self.state[0]) ):
                if(self.state[i-1][j+1]!=3):
                    immediate_neighbors.append((i-1,j+1))
            if ( i < len(self.state) and j + 1 < len(self.state[0]) ):
                if(self.state[i][j+1]!=3):
                    immediate_neighbors.append((i,j+1))
            if ( i + 1 < len(self.state) and j < len(self.state[0]) ):
                if(self.state[i+1][j]!=3):
                    immediate_neighbors.append((i+1,j))
            if ( i < len(self.state) and j - 1 >= 0 ):
                if(self.state[i][j-1]!=3):
                    immediate_neighbors.append((i,j-1))
            if ( i - 1 >= 0 and j - 1 >= 0 ):
                if(self.state[i-1][j-1]!=3):
                    immediate_neighbors.append((i-1,j-1))
            
        return immediate_neighbors
    def neither_edge_nor_corner(self,action) -> bool:
        for i in range(len(self.edge_cells)):
            if(action[0] == self.edge_cells[i][0] and action[1]==self.edge_cells[i][1]):
                return True
        for i in range(len(self.corner_cells)):
            if(action[0] == self.corner_cells[i][0] and action[1]==self.corner_cells[i][1]):
                return True
        return False
    def identify_dead_cells(self) :
        dead_cells = []
        for action in self.possible_actions:
            if ( self.neither_edge_nor_corner(action) == False ): 
                neighbor = self.get_immediate_connections(action[0],action[1])
                #print(neighbor)
                action_dead = False
                
                for i in range(len(neighbor)):
                    if(action_dead == True):    
                        break

                    if ( self.state[neighbor[i][0]][neighbor[i][1]] ==1 and self.state[neighbor[(i+1)%6][0]][neighbor[(i+1)%6][1]] == 1  and self.state[neighbor[(i+2)%6][0]][neighbor[(i+2)%6][1]] == 1 and self.state[neighbor[(i+3)%6][0]][neighbor[(i+3)%6][1]] == 1 ):
                        
                        action_dead = True
                        break
                    elif( self.state[neighbor[i][0]][neighbor[i][1]] ==2 and self.state[neighbor[(i+1)%6][0]][neighbor[(i+1)%6][1]] == 2  and self.state[neighbor[(i+2)%6][0]][neighbor[(i+2)%6][1]] == 2 and self.state[neighbor[(i+3)%6][0]][neighbor[(i+3)%6][1]] == 2 ):
                       
                       
                        action_dead = True

                        break
                    elif (self.state[neighbor[i][0]][neighbor[i][1]] ==2 and self.state[neighbor[(i+1)%6][0]][neighbor[(i+1)%6][1]] == 2  and self.state[neighbor[(i+2)%6][0]][neighbor[(i+2)%6][1]] == 2 and self.state[neighbor[(i+4)%6][0]][neighbor[(i+4)%6][1]] == 1 ):

                        action_dead = True

                        break
                    elif(self.state[neighbor[i][0]][neighbor[i][1]] ==1 and self.state[neighbor[(i+1)%6][0]][neighbor[(i+1)%6][1]] == 1  and self.state[neighbor[(i+2)%6][0]][neighbor[(i+2)%6][1]] == 1 and self.state[neighbor[(i+4)%6][0]][neighbor[(i+4)%6][1]] == 2 ):

                        action_dead = True

                        break
                    elif(self.state[neighbor[i][0]][neighbor[i][1]] ==1 and self.state[neighbor[(i+1)%6][0]][neighbor[(i+1)%6][1]] == 1  and self.state[neighbor[(i+3)%6][0]][neighbor[(i+3)%6][1]] == 2 and self.state[neighbor[(i+4)%6][0]][neighbor[(i+4)%6][1]] == 2 ):

                        action_dead = True

                        break
                    
                if(action_dead):

                    r = ( action[0],action[1] )
                    
                    dead_cells += [r]
                    continue
   
        return  dead_cells
    def adjust_time_limit(self,remaining_time): 
        self.time_limit = 1
        return 
    def MCTS_search(self,action):
        v = 0

        self.state[action[0]][action[1]] = self.player_number
        self.union_find(action[0],action[1],self.player_number)
        self.my_last_move = action
        if action in self.possible_actions:
            self.possible_actions.remove(action)
        factor = 0.5
        threshold = 0.25
        game_ended = False
        if(self.possible_actions == []):
            game_ended = True
        scored_on_this = 0 
        while(not game_ended):

            player2_move,winning_move = self.best_next_move_for_opponent()
            if(winning_move):
                game_ended = True
                break
            if(player2_move[0]==-1 and player2_move[1]==-1):
                player2_move,scored_on_this = self.get_closest_next_moves(3-self.player_number)[0]
            self.state[player2_move[0]][player2_move[1]] = 3-self.player_number
            self.union_find(player2_move[0],player2_move[1],3-self.player_number)
            self.last_move = player2_move
            
            if(self.possible_actions == []):
                game_ended = True
                break
            if player2_move in self.possible_actions:
                self.possible_actions.remove(player2_move)


            player1_move,winning_move = self.best_next_move_for_opponent()
            if(winning_move):
                return 100*factor  
              
            if(player1_move[0]==-1 and player1_move[1]==-1):
                player1_move,scored_on_this = self.get_closest_next_moves(self.player_number)[0]
            v += scored_on_this*factor
            self.state[player1_move[0]][player1_move[1]] = self.player_number
            self.union_find(player1_move[0],player1_move[1],self.player_number)
            self.my_last_move = player1_move
            if player1_move in self.possible_actions:
                self.possible_actions.remove(player1_move)

            if factor < threshold :
                w =  self.get_closest_next_moves(self.player_number)[0][1]

                return ( w*factor + v )
            
            factor*=0.5
            if(self.possible_actions == []):
                game_ended = True
                break
            
        


        return v
    def MCTS(self):
        #return max(self.possible_actions,key = lambda x : (self.evaluate_state(self.state,x,self.player_number)))
        
        n = len(self.state)
        m = len(self.state[1])


        store_state = np.zeros((n,m))
        store_union_connections = np.zeros((n*m,4),dtype = int)
        store_possible_actions = []
        store_last_move = self.last_move
        store_my_last_move = self.my_last_move
        maximum_move = (-1,-1)
        maximum_heuristic_value = -10000000
        for  i in range(n):
            for j in range(m):
                
                store_state[i][j] = self.state[i][j]
                store_union_connections[i*len(self.state)+j][0] = self.union_connections[i*len(self.state)+j][0]
                store_union_connections[i*len(self.state)+j][1] = self.union_connections[i*len(self.state)+j][1]
                store_union_connections[i*len(self.state)+j][2] = self.union_connections[i*len(self.state)+j][2]
                store_union_connections[i*len(self.state)+j][3] = self.union_connections[i*len(self.state)+j][3] 
        for action in self.possible_actions:
            store_possible_actions.append(action)     
        
        

        start_time = self.start_time
        move_values = []
        while( math.floor(time.time() - start_time) < self.time_limit):
            best_possible_moves = self.get_closest_next_moves(self.player_number)
            for (action_number,value_of_move) in best_possible_moves:
                
                value = self.MCTS_search(action_number)+value_of_move

                
                self.possible_actions = []
                self.my_last_move = store_my_last_move
                self.last_move = store_last_move
                
                for  i in range(n):
                    for j in range(m):
                        self.state[i][j] = store_state[i][j]
                        self.union_connections[i*len(self.state)+j][0] = store_union_connections[i*len(self.state)+j][0]
                        self.union_connections[i*len(self.state)+j][1] = store_union_connections[i*len(self.state)+j][1]
                        self.union_connections[i*len(self.state)+j][2] = store_union_connections[i*len(self.state)+j][2]
                        self.union_connections[i*len(self.state)+j][3] = store_union_connections[i*len(self.state)+j][3]
                for action in store_possible_actions:
                    self.possible_actions.append(action)
                #print("MCTS move ",self.possible_actions[action_number]," , ",value)
                move_values.append((action_number,value))


        
        self.possible_actions = []
        self.my_last_move = store_my_last_move
        self.last_move = store_last_move
        
        for  i in range(n):
            for j in range(m):
                self.state[i][j] = store_state[i][j]
                self.union_connections[i*len(self.state)+j][0] = store_union_connections[i*len(self.state)+j][0]
                self.union_connections[i*len(self.state)+j][1] = store_union_connections[i*len(self.state)+j][1]
                self.union_connections[i*len(self.state)+j][2] = store_union_connections[i*len(self.state)+j][2]
                self.union_connections[i*len(self.state)+j][3] = store_union_connections[i*len(self.state)+j][3]
        for action in store_possible_actions:
            self.possible_actions.append(action)
        


        for i in range(len(move_values)):
            if( (move_values[i][1] + self.evaluate_state(move_values[i][0],self.player_number)*2)>maximum_heuristic_value):
                maximum_heuristic_value = (move_values[i][1] + self.evaluate_state(move_values[i][0],self.player_number)*2)
                maximum_move = move_values[i][0]

        return maximum_move
    def get_move(self, state: np.array) -> Tuple[int, int]:
        self.start_time = time.time()    
        r = (0,0)

        if(self.game_setup_initialised == False and self.player_number == 1):
            
            self.Initialise_game(state, self.player_number)
            self.game_setup_initialised = True
            #print(self.corner_cells,self.edge_cells)
            r = (0,0)
            self.my_last_move = r
            if r in self.possible_actions:
                self.possible_actions.remove(r)

            self.state[r[0]][r[1]] = self.player_number
            self.union_find(r[0],r[1],self.player_number)
            #print(self.state)
            # for i in range(len(self.state)):
            #     for j in range (len(self.state[0])):
            #         print(" (",self.union_connections[i*len(self.state)+j][0],",",self.union_connections[i*len(self.state)+j][1],",", self.union_connections[i*len(self.state)+j][2],",",self.union_connections[i*len(self.state)+j][3],") ",end=" ")
            #     print()
            return r

        elif (self.game_setup_initialised == False and self.player_number == 2):
            self.Initialise_game(state, self.player_number)
            

            self.game_setup_initialised = True
            r = (0,0)
            if(self.last_move==(0,0)):
                r = (0,self.game_size -1)
            self.my_last_move = r
            if r in self.possible_actions:
                self.possible_actions.remove(r)
            self.state[r[0]][r[1]] = self.player_number
            self.union_find(r[0],r[1],self.player_number)
            #print(self.state)
            # for i in range(len(self.state)):
            #     for j in range (len(self.state[0])):
            #         print(" (",self.union_connections[i*len(self.state)+j][0],",",self.union_connections[i*len(self.state)+j][1],",", self.union_connections[i*len(self.state)+j][2],",",self.union_connections[i*len(self.state)+j][3],") ",end=" ")
            #     print()
            return r
            

            #print(self.corner_cells,self.edge_cells)
        else:
            self.get_opponent_move(state)


            if(self.possible_actions == []):
                self.possible_actions = self.dead_cells
            

            
            self.adjust_time_limit(fetch_remaining_time(self.timer, self.player_number))
            


            #self.MCTS()
            r,Winned =  self.best_next_move_for_me()
            if (r[0]==-1 and r[1]==-1):
                #print("doing MCTS")
                r = self.MCTS()
            #r= random.choice(self.possible_actions)
            self.my_last_move = r
            if r in self.possible_actions:
                self.possible_actions.remove(r)

            #print("AI move ",r)
            self.state[r[0]][r[1]] = self.player_number
            self.union_find(r[0],r[1],self.player_number)
            #print(self.state)
            # for i in range(len(self.state)):
            #     for j in range (len(self.state[0])):
            #         print(" (" , i*len(self.state)+j," -> ",self.union_connections[i*len(self.state)+j][0],",",self.union_connections[i*len(self.state)+j][1],",",self.union_connections[i*len(self.state)+j][2],",",self.union_connections[i*len(self.state)+j][3],") ",end=" ")
            #     print()
            return r
        
        return random.choice(self.possible_actions)
        raise NotImplementedError('Whoops I don\'t know what to do')

    
    