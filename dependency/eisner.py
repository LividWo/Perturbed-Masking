import numpy as np


class Eisner(object):
    """
    Dependency decoder class
    """

    def __init__(self):
        self.verbose = False

    def parse_proj(self, scores):
        """
        Parse using Eisner's algorithm.
        """

        # ----------
        # Solution to Exercise 4.3.6
        nr, nc = np.shape(scores)
        if nr != nc:
            raise ValueError("scores must be a squared matrix with nw+1 rows")
            return []

        N = nr - 1  # Number of words (excluding root).

        # Initialize CKY table.
        complete = np.zeros([N+1, N+1, 2])  # s, t, direction (right=1).
        incomplete = np.zeros([N+1, N+1, 2])  # s, t, direction (right=1).
        complete_backtrack = -np.ones([N+1, N+1, 2], dtype=int)  # s, t, direction (right=1).
        incomplete_backtrack = -np.ones([N+1, N+1, 2], dtype=int)  # s, t, direction (right=1).

        incomplete[0, :, 0] -= np.inf

        # Loop from smaller items to larger items.
        for k in range(1, N+1):
            for s in range(N-k+1):
                t = s + k

                # First, create incomplete items.
                # left tree
                incomplete_vals0 = complete[s, s:t, 1] + complete[(s+1):(t+1), t, 0] + scores[t, s]
                incomplete[s, t, 0] = np.max(incomplete_vals0)
                incomplete_backtrack[s, t, 0] = s + np.argmax(incomplete_vals0)
                # right tree
                incomplete_vals1 = complete[s, s:t, 1] + complete[(s+1):(t+1), t, 0] + scores[s, t]
                incomplete[s, t, 1] = np.max(incomplete_vals1)
                incomplete_backtrack[s, t, 1] = s + np.argmax(incomplete_vals1)

                # Second, create complete items.
                # left tree
                complete_vals0 = complete[s, s:t, 0] + incomplete[s:t, t, 0]
                complete[s, t, 0] = np.max(complete_vals0)
                complete_backtrack[s, t, 0] = s + np.argmax(complete_vals0)
                # right tree
                complete_vals1 = incomplete[s, (s+1):(t+1), 1] + complete[(s+1):(t+1), t, 1]
                complete[s, t, 1] = np.max(complete_vals1)
                complete_backtrack[s, t, 1] = s + 1 + np.argmax(complete_vals1)

        value = complete[0][N][1]
        heads = -np.ones(N + 1, dtype=int)
        self.backtrack_eisner(incomplete_backtrack, complete_backtrack, 0, N, 1, 1, heads)

        value_proj = 0.0
        for m in range(1, N+1):
            h = heads[m]
            value_proj += scores[h, m]

        return heads, value_proj

        # End of solution to Exercise 4.3.6
        # ----------

    def backtrack_eisner(self, incomplete_backtrack, complete_backtrack, s, t, direction, complete, heads):
        """
        Backtracking step in Eisner's algorithm.
        - incomplete_backtrack is a (NW+1)-by-(NW+1) numpy array indexed by a start position,
        an end position, and a direction flag (0 means left, 1 means right). This array contains
        the arg-maxes of each step in the Eisner algorithm when building *incomplete* spans.
        - complete_backtrack is a (NW+1)-by-(NW+1) numpy array indexed by a start position,
        an end position, and a direction flag (0 means left, 1 means right). This array contains
        the arg-maxes of each step in the Eisner algorithm when building *complete* spans.
        - s is the current start of the span
        - t is the current end of the span
        - direction is 0 (left attachment) or 1 (right attachment)
        - complete is 1 if the current span is complete, and 0 otherwise
        - heads is a (NW+1)-sized numpy array of integers which is a placeholder for storing the
        head of each word.
        """
        if s == t:
            return
        if complete:
            r = complete_backtrack[s][t][direction]
            if direction == 0:
                self.backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 0, 1, heads)
                self.backtrack_eisner(incomplete_backtrack, complete_backtrack, r, t, 0, 0, heads)
                return
            else:
                self.backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 1, 0, heads)
                self.backtrack_eisner(incomplete_backtrack, complete_backtrack, r, t, 1, 1, heads)
                return
        else:
            r = incomplete_backtrack[s][t][direction]
            if s == 1 or t == 1:
                pass
            if direction == 0:
                heads[s] = t
                self.backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 1, 1, heads)
                self.backtrack_eisner(incomplete_backtrack, complete_backtrack, r+1, t, 0, 1, heads)
                return
            else:
                heads[t] = s
                self.backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 1, 1, heads)
                self.backtrack_eisner(incomplete_backtrack, complete_backtrack, r+1, t, 0, 1, heads)
                return


# ***************************************************************
if __name__ == '__main__':
    # 7x7 matrix
    scores = [
        [0.0, 0.21789748606009915, 0.2285407168033741, 0.20008290881922605, 0.1784354074121896, 0.14371876134812747,
         0.1649149059628676],
        [0.1666260891011342, 0.0, 0.23763878138999808, 0.06686100500754769, 0.04252177163502459, 0.05377208099090549,
         0.07318417053825277],
        [0.14819371521141986, 0.2561611006628804, 0.0, 0.22763372573718102, 0.09078768244051524, 0.10544933293378421,
         0.06696645233053426],
        [0.33278193213153157, 0.24182637172052882, 0.24348700794789607, 0.0, 0.3074931499229424, 0.34207021338827026,
         0.38077567455395994],
        [0.12199173439186563, 0.0856421256888373, 0.0856836501258372, 0.21524582195485076, 0.0, 0.2090616298352853,
         0.1473791651283505],
        [0.14103502167319285, 0.1383250368740608, 0.13255458167128512, 0.16840110553265578, 0.25125240895112677, 0.0,
         0.16677963148603495],
        [0.08937150749085589, 0.0601478789935935, 0.07209526206160943, 0.12177543294853871, 0.12950957963820142,
         0.14592798150362726, 0.0]
    ]
    scores = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    decoder = Eisner()
    scores = np.array(scores)
    best_arcs, root_pred = decoder.parse_proj_no_root(scores)