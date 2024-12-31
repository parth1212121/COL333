import heapq
from collections import defaultdict

class Agent(object):
    def __init__(self, phoneme_table, vocabulary) -> None:
        """
        Agent initialization. You can also add code but don't remove the existing code.
        """
        self.phoneme_table = phoneme_table
        self.word_phoneme_table = []
        
        self.vocabulary = vocabulary
        self.best_state = None
        self.replacement_history = defaultdict(lambda: defaultdict(dict))
        self.flag_end_words = set()
        self.flag_start_words = set()
        self.visited_states = set()
        self.count_total_states = 0

    def update_phoneme_table(self, environment):
        
        keys = []
        for key in self.phoneme_table.keys():
            keys.append(key)
        self.word_phoneme_table.clear()
        self.word_phoneme_table = []
        words = environment.init_state.split()

        for dd in range(0,len(words)):
          
            default_dict = dict()
            for key in keys:
                
                default_dict[key] = []
                for val in self.phoneme_table[key]:

                    if words[dd].find(val) != -1:
                        default_dict[key] += [ val]
                    
                if(len(default_dict[key])==0):
                    default_dict.pop(key)
            self.word_phoneme_table.append(default_dict)
            
        # for i in range(0,len(words)):
        #     print(words[i] , " -> ")
        #     for key in self.word_phoneme_table[i].keys():
        #         print(key , " : " , self.word_phoneme_table[i][key])
       

   
    def generate_candidates(self, state, environment, best_cost, cost_threshold, added_front=False, added_back=False):

        candidates = []
        words = state.split()
        parent_cost = environment.compute_cost(state)
        x = -1
        remove_states = []
        for i, word in enumerate(words):

            if added_front and i==0:
                continue
            if added_back and i==(len(words)-1):
                continue
            x+=1
            for substitute, phonemes in self.word_phoneme_table[x].items():
                for phoneme in phonemes:
                    start = 0
                    while start < len(word):
                        index = word.find(phoneme, start)
                        if index == -1:
                            break

                        before_len = index
                        after_len = len(word) - index - len(phoneme)
                        if (before_len, after_len) in self.replacement_history[word][(substitute, phoneme)]:
                            if self.replacement_history[word][(substitute, phoneme)][(before_len, after_len)]:
                                #print(f"  Skipping in flagged: Word='{word}', Substitute='{substitute}', Phoneme='{phoneme}', Pos={index}")
                                start = index + 1
                                continue  

                        new_word = word[:index] + substitute + word[index + len(phoneme):]
                        new_state = " ".join(words[:i] + [new_word] + words[i+1:])
                        if new_state in self.visited_states:
                        #     #print(f"  Skipping already visited state: {new_state}")
                            start = index + 1
                            continue  

                        self.count_total_states += 1
                        cost = environment.compute_cost(new_state)

                        if cost <= parent_cost * cost_threshold:
                            candidates.append((new_state, cost, added_front, added_back))
                            self.replacement_history[word][(substitute, phoneme)][(before_len, after_len)] = False 

                            #print(f"  Added candidate: {new_state} with cost: {cost}")
                        else:

                            #candidates.append((new_state, cost, added_front, added_back))
                            self.replacement_history[word][(substitute, phoneme)][(before_len, after_len)] = True
         
                            #print(f"  Flagged candidate as unhelpful: {new_state} with cost: {cost}")
                            #print(f"  ADDED VISITED : Word='{word}', Substitute='{substitute}', Phoneme='{phoneme}', Pos={index}")

                        start = index + 1


        if not added_front:
            for word in self.vocabulary:
                if word not in self.flag_start_words:
                    new_state = word + " " + state
                    if new_state in self.visited_states:
                    #     #print(f"  Skipping already visited state (word at front): {new_state}")
                        continue  
                    self.count_total_states += 1
                    cost = environment.compute_cost(new_state)
                    if cost <= parent_cost * cost_threshold:
                        candidates.append((new_state, cost, True, added_back))
                        #print(f"  Added candidate (word at front): {new_state} with cost: {cost}")
                    else:
                        #print(f"  Added visited (word at front): {new_state} with cost: {cost}")
                        self.flag_start_words.add(word)

        if not added_back:
            for word in self.vocabulary:
                if word not in self.flag_end_words:
                    new_state = state + " " + word
                    if new_state in self.visited_states:
                    #     #print(f"  Skipping already visited state (word at back): {new_state}")
                        continue  # Skip if the state has already been visited
                    self.count_total_states += 1
                    cost = environment.compute_cost(new_state)
                    if cost <= parent_cost * cost_threshold:
                        candidates.append((new_state, cost, added_front, True))
                        #print(f"  Added candidate (word at back): {new_state} with cost: {cost}")
                    else:
                        #print(f"  Added visited (word at front): {new_state} with cost: {cost}")
                        self.flag_end_words.add(word)

        return candidates
    

    def local_beam_search_dynamic(self, environment, initial_beam_width=5, min_beam_width=2, max_iterations=5, cost_threshold=1.1):
        """
        Local Beam Search implementation with Dynamic Beam Width, Pruning, and State Tracking for replacements.
        """
        self.best_state = environment.init_state
        best_cost = environment.compute_cost(self.best_state)
        #print(f"Initial state: {self.best_state}, Initial cost: {best_cost}")
        #self.remove_non_profitable_substitutions(environment,best_cost)

        beam_width = initial_beam_width
        beam = [(self.best_state, best_cost, False, False)]
        self.visited_states.add(self.best_state)  
        iterations = 0
        

        while beam and iterations < max_iterations:
            # print(f"\nIteration: {iterations} ")
            all_candidates = []
            for state, cost, added_front, added_back in beam:
                #print(f"Expanding state: {state} with cost: {cost}")
                candidates = self.generate_candidates(state, environment, best_cost, cost_threshold, added_front, added_back)
                for candidate in candidates:
                    new_state, new_cost, _, _ = candidate
                    if new_state not in self.visited_states:
                        all_candidates.append(candidate)
                        self.visited_states.add(new_state)  

            if(self.visited_states.__sizeof__() > 1000000):
                self.visited_states.clear()
                self.visited_states.add(new_state)

                # for candidate in candidates:
                #     new_state, new_cost, _, _ = candidate
                #     # print(f"Neighbour state: {new_state} with cost: {new_cost}")
                #     # if new_state not in self.visited_states:
                #     #     all_candidates.append(candidate)
                #     #     self.visited_states.add(new_state)             


            beam = heapq.nsmallest(beam_width, all_candidates, key=lambda x: x[1])

            # print("\nCurrent beam:")
            # for state, cost, _, _ in beam:
                # print(f"  State: {state}, Cost: {cost}")

            for state, cost, _, _ in beam:
                if cost < best_cost:
                    best_cost = cost
                    self.best_state = state
                    #print(f"  New best state: {self.best_state} with cost: {best_cost}")

            # print("Present_iterations ->", iterations)


            beam_width = max(min_beam_width, int(beam_width * 0.7))
            iterations += 1

        # print(f"\nFinal best state: {self.best_state}, Final best cost: {best_cost}")

    def asr_corrector(self, environment):
        """
        Your ASR corrector agent goes here. You can apply multiple strategies
        here by calling the respective functions.
        """
        self.visited_states.clear()
        self.flag_end_words.clear()
        self.flag_start_words.clear()
        self.update_phoneme_table(environment)
        self.count_total_states = 0
        self.replacement_history.clear()
        self.local_beam_search_dynamic(environment, initial_beam_width=6, min_beam_width=4, max_iterations=15, cost_threshold=1.1)
        
        #open daksh_answer.json file and compare ith line from the list with the best_state

        # print("final size of visited states :", len(self.visited_states))
        # print("total states called :", self.count_total_states)

        #open daksh_answer.json file and compare ith line from the list with the best_state
        # pf = open('answer.json', 'r')
        # lines = pf.readlines()
        # line = lines[line_number]
        # line = line.strip()
        # if ("\"" +self.best_state + "\",") ==  line :
        #     print("Correct")
        # else:
        #     print("Incorrect")
        #     print("Expected :", line)
        # print()
        # print( " ---------------------------")
        # print()
        environment.best_state = self.best_state
