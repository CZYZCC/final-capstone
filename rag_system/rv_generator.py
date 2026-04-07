import random
import math
from collections import deque
from typing import Dict, Any, List


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class RVQuestion:
    """Every RV question class inherits from this."""

    def initialization(self):
        raise NotImplementedError

    def solution(self) -> Any:
        raise NotImplementedError

    def generation(self) -> Dict[str, Any]:
        """
        Returns a dict with at minimum:
          question  : str   — the question text
          answer    : str   — the verified correct answer (computed by code)
          algorithm : str   — algorithm name (used for distractor generation)
          meta      : dict  — raw values for distractor generation prompt
        """
        raise NotImplementedError

    def sample(self) -> Dict[str, Any]:
        """Convenience: run all three steps and return the full QA dict."""
        self.initialization()
        return self.generation()


# ---------------------------------------------------------------------------
# 1. Hash Table — Linear Probing
# ---------------------------------------------------------------------------

class HashTableLinearProbingQuestion(RVQuestion):
    """
    Edge-case design: initialization() deliberately generates keys that all
    share the same initial hash slot, forcing a maximum-length probe chain.
    """

    def initialization(self):
        self.m = random.choice([5, 7, 10, 11])
        # Choose a base residue — all "collider" keys map here
        base_residue = random.randint(0, self.m - 1)
        n_colliders  = random.randint(3, min(5, self.m - 1))
        # Collider keys: base_residue, base_residue+m, base_residue+2m, ...
        self.keys = [base_residue + i * self.m for i in range(n_colliders)]
        # Add 1–2 non-colliding keys for variety
        n_extra = random.randint(1, 2)
        extra = []
        for _ in range(n_extra * 10):
            k = random.randint(1, 50)
            if k % self.m != base_residue and k not in self.keys:
                extra.append(k)
            if len(extra) == n_extra:
                break
        # Insert non-colliders first, then the collision chain (harder)
        random.shuffle(extra)
        self.keys = extra + self.keys
        self.query_key = self.keys[-1]  # ask about the last inserted (most probes)

    def solution(self) -> int:
        table = [None] * self.m
        for key in self.keys:
            pos = key % self.m
            while table[pos] is not None:
                pos = (pos + 1) % self.m
            table[pos] = key
        for i, v in enumerate(table):
            if v == self.query_key:
                return i
        return -1  # should never happen

    def generation(self) -> Dict[str, Any]:
        answer = self.solution()
        question = (
            f"A hash table of size {self.m} uses linear probing with "
            f"h(k) = k mod {self.m}. Insert keys {self.keys} in order. "
            f"After all insertions, what is the final index (0-based) of key {self.query_key}?"
        )
        # Build wrong-answer candidates (for distractor generation prompt)
        initial_slot = self.query_key % self.m
        return {
            "question":  question,
            "answer":    str(answer),
            "algorithm": "hash table linear probing",
            "meta": {
                "table_size":    self.m,
                "keys":          self.keys,
                "query_key":     self.query_key,
                "initial_slot":  initial_slot,
                "correct_index": answer,
                "common_wrong": [
                    initial_slot,                      # no probing at all
                    (initial_slot + 1) % self.m,       # one probe too few
                    (answer + 1) % self.m,             # one probe too many
                ],
            },
        }


# ---------------------------------------------------------------------------
# 2. Hash Table — Separate Chaining
# ---------------------------------------------------------------------------

class HashTableChainingQuestion(RVQuestion):
    """Ask about chain length in a specific bucket after all insertions."""

    def initialization(self):
        self.m = random.choice([5, 7, 10])
        # All keys hash to bucket 0 — maximises chain length
        n_keys = random.randint(4, 7)
        self.keys = [i * self.m for i in range(1, n_keys + 1)]
        random.shuffle(self.keys)
        self.query_bucket = 0

    def solution(self) -> int:
        chains = {i: [] for i in range(self.m)}
        for k in self.keys:
            chains[k % self.m].append(k)
        return len(chains[self.query_bucket])

    def generation(self) -> Dict[str, Any]:
        answer = self.solution()
        question = (
            f"A hash table with {self.m} buckets uses separate chaining "
            f"with h(k) = k mod {self.m}. "
            f"Insert keys {self.keys} in order. "
            f"What is the chain length in bucket {self.query_bucket} after all insertions?"
        )
        return {
            "question":  question,
            "answer":    str(answer),
            "algorithm": "hash table separate chaining",
            "meta": {
                "m":            self.m,
                "keys":         self.keys,
                "bucket":       self.query_bucket,
                "chain_length": answer,
            },
        }


# ---------------------------------------------------------------------------
# 3. Insertion Sort
# ---------------------------------------------------------------------------

class InsertionSortQuestion(RVQuestion):
    """
    Edge-case: nearly-sorted array with one element severely out of place,
    maximising the number of shifts for that element.
    """

    def initialization(self):
        n = random.randint(5, 7)
        # Nearly sorted: [2,3,4,5,6] with 1 inserted at the end
        sorted_part = random.sample(range(2, 20), n - 1)
        sorted_part.sort()
        outlier = random.randint(sorted_part[0] - 5, sorted_part[0] - 1)
        outlier = max(1, outlier)
        self.arr = sorted_part + [outlier]   # outlier at the end → max shifts
        self.variant = random.choice(["sorted_array", "number_of_comparisons"])

    def solution(self):
        a = self.arr[:]
        comparisons = 0
        for i in range(1, len(a)):
            key = a[i]
            j = i - 1
            while j >= 0 and a[j] > key:
                comparisons += 1
                a[j + 1] = a[j]
                j -= 1
            if j >= 0:
                comparisons += 1  # the failing comparison
            a[j + 1] = key
        return {"sorted": a, "comparisons": comparisons}

    def generation(self) -> Dict[str, Any]:
        sol = self.solution()
        if self.variant == "sorted_array":
            question = (
                f"Apply insertion sort to the array {self.arr}. "
                f"What is the final sorted array?"
            )
            answer = str(sol["sorted"])
        else:
            question = (
                f"Apply insertion sort to the array {self.arr}. "
                f"How many key comparisons are made in total?"
            )
            answer = str(sol["comparisons"])
        return {
            "question":  question,
            "answer":    answer,
            "algorithm": "insertion sort",
            "meta":      {"arr": self.arr, "variant": self.variant, **sol},
        }


# ---------------------------------------------------------------------------
# 4. Bubble Sort — Stability with Duplicate Keys
# ---------------------------------------------------------------------------

class BubbleSortStabilityQuestion(RVQuestion):
    """
    Edge-case: array with duplicate values to test stability understanding.
    Question: how many swaps occur in the first pass?
    """

    def initialization(self):
        n = random.randint(5, 6)
        pool = list(range(1, 8))
        # Force at least 2 duplicates
        self.arr = random.choices(pool, k=n)
        while len(set(self.arr)) >= n - 1:
            self.arr = random.choices(pool, k=n)

    def solution(self):
        a = self.arr[:]
        swaps_pass1 = 0
        total_passes = 0
        for i in range(len(a)):
            swapped = False
            for j in range(len(a) - i - 1):
                if a[j] > a[j + 1]:
                    a[j], a[j + 1] = a[j + 1], a[j]
                    if i == 0:
                        swaps_pass1 += 1
                    swapped = True
            if not swapped:
                total_passes = i + 1
                break
        else:
            total_passes = len(a)
        return {"sorted": a, "swaps_pass1": swaps_pass1, "passes": total_passes}

    def generation(self) -> Dict[str, Any]:
        sol = self.solution()
        question = (
            f"Apply bubble sort to the array {self.arr}. "
            f"How many swaps occur during the FIRST pass over the array?"
        )
        return {
            "question":  question,
            "answer":    str(sol["swaps_pass1"]),
            "algorithm": "bubble sort",
            "meta":      {"arr": self.arr, **sol},
        }


# ---------------------------------------------------------------------------
# 5. Merge Sort — State After First Merge Level
# ---------------------------------------------------------------------------

class MergeSortQuestion(RVQuestion):
    """Ask for the array state after all level-1 merges (pairs of 1-element lists)."""

    def initialization(self):
        n = random.choice([4, 6, 8])
        self.arr = random.sample(range(1, 30), n)
        # Guarantee at least one adjacent pair is in REVERSE order so
        # level-1 merges actually do something visible (edge case triggered).
        # Pick a random even index and swap the pair if it's already sorted.
        even_indices = list(range(0, n - 1, 2))
        random.shuffle(even_indices)
        for i in even_indices:
            if self.arr[i] < self.arr[i + 1]:   # pair is in order → reverse it
                self.arr[i], self.arr[i + 1] = self.arr[i + 1], self.arr[i]
                break

    def _merge(self, left, right):
        result = []
        i = j = 0
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                result.append(left[i]); i += 1
            else:
                result.append(right[j]); j += 1
        return result + left[i:] + right[j:]

    def solution(self):
        # After level-1 merges: sort adjacent pairs
        a = self.arr[:]
        after_level1 = []
        for i in range(0, len(a) - 1, 2):
            pair = self._merge([a[i]], [a[i + 1]])
            after_level1.extend(pair)
        if len(a) % 2 == 1:
            after_level1.append(a[-1])
        final = sorted(self.arr)  # full sort for comparison
        return {"after_level1": after_level1, "final": final}

    def generation(self) -> Dict[str, Any]:
        sol = self.solution()
        question = (
            f"Apply merge sort (bottom-up) to the array {self.arr}. "
            f"What is the array after completing ALL level-1 merges "
            f"(i.e., after merging every adjacent pair of single elements)?"
        )
        return {
            "question":  question,
            "answer":    str(sol["after_level1"]),
            "algorithm": "merge sort",
            "meta":      {"arr": self.arr, **sol},
        }


# ---------------------------------------------------------------------------
# 6. BST — Insert + Successor
# ---------------------------------------------------------------------------

class BSTOperationsQuestion(RVQuestion):
    """
    Edge-case: degenerate insertion order (descending) → linear left chain.
    Question: inorder traversal, height, or successor of a given node.
    """

    def initialization(self):
        n = random.randint(5, 7)
        # Descending insertion → degenerate left-skewed tree
        base = random.randint(40, 80)          # high enough that all keys stay positive
        step = random.randint(3, 8)
        self.keys = [base - i * step for i in range(n)]
        # Safety: all keys must be positive
        if self.keys[-1] <= 0:
            self.keys = [k + abs(self.keys[-1]) + 5 for k in self.keys]
        # "inorder" is REMOVED: inorder of any BST = sorted(keys) — trivially easy,
        # GRD=2 always.  "successor_after_search" chains TWO operations causally
        # (search result → successor lookup), earning GRD=4.
        self.variant = random.choice(["height", "search_steps", "successor_after_search"])
        if self.variant == "search_steps":
            # Search for a key NOT in the tree
            self.search_key = self.keys[n // 2] - 1  # between two existing keys
        elif self.variant == "successor_after_search":
            # Search for an existing key, then ask for its inorder successor
            self.search_key = random.choice(self.keys[1:-1])  # not root, not min

    class _Node:
        def __init__(self, key):
            self.key = key
            self.left = self.right = None

    def _insert(self, root, key):
        if root is None:
            return self._Node(key)
        if key < root.key:
            root.left  = self._insert(root.left,  key)
        elif key > root.key:
            root.right = self._insert(root.right, key)
        return root

    def _height(self, node):
        if node is None: return 0
        return 1 + max(self._height(node.left), self._height(node.right))

    def _inorder(self, node, result):
        if node is None: return
        self._inorder(node.left,  result)
        result.append(node.key)
        self._inorder(node.right, result)

    def _search_steps(self, root, key):
        steps = 0
        node = root
        while node is not None:
            steps += 1
            if key == node.key:
                return steps, True
            elif key < node.key:
                node = node.left
            else:
                node = node.right
        return steps, False

    def _find_successor(self, root, key):
        """Return the inorder successor key of `key` in BST rooted at `root`."""
        successor = None
        node = root
        while node is not None:
            if key < node.key:
                successor = node.key   # candidate: this ancestor is > key
                node = node.left
            elif key > node.key:
                node = node.right
            else:
                # Found the node; if right subtree exists, successor = min there
                if node.right:
                    n = node.right
                    while n.left:
                        n = n.left
                    return n.key
                break
        return successor   # highest ancestor that we went left from

    def solution(self):
        root = None
        for k in self.keys:
            root = self._insert(root, k)
        if self.variant == "height":
            return {"height": self._height(root)}
        elif self.variant == "search_steps":
            steps, found = self._search_steps(root, self.search_key)
            return {"steps": steps, "found": found}
        else:   # successor_after_search
            successor = self._find_successor(root, self.search_key)
            return {"successor": successor}

    def generation(self) -> Dict[str, Any]:
        sol = self.solution()
        if self.variant == "height":
            question = (
                f"Insert keys {self.keys} one by one into an initially empty BST "
                f"(duplicates are not allowed; use standard comparison). "
                f"What is the height of the resulting tree?"
            )
            answer = str(sol["height"])
        elif self.variant == "search_steps":
            question = (
                f"Insert keys {self.keys} one by one into an initially empty BST. "
                f"Search for key {self.search_key} (which is NOT in the tree). "
                f"How many node comparisons are made before the search terminates?"
            )
            answer = str(sol["steps"])
        else:   # successor_after_search — two-step causal chain for GRD=4
            question = (
                f"Insert keys {self.keys} one by one into an initially empty BST "
                f"(duplicates not allowed). "
                f"First, search for key {self.search_key} (which IS in the tree) "
                f"and locate its node. "
                f"Then find the inorder successor of that node. "
                f"What is the key of the inorder successor?"
            )
            answer = str(sol["successor"])
        return {
            "question":  question,
            "answer":    answer,
            "algorithm": "binary search tree",
            "meta":      {"keys": self.keys, "variant": self.variant, **sol},
        }


# ---------------------------------------------------------------------------
# 7. BFS — Connected Components
# ---------------------------------------------------------------------------

class BFSComponentsQuestion(RVQuestion):
    """
    Edge-case: graph with 1 large component, 1 small component, and 1 isolated node.
    Question: number of connected components, or BFS visit order from a start node.
    """

    def initialization(self):
        n = random.randint(6, 8)
        k1 = random.randint(3, 4)
        self.n = n
        edges = []
        for i in range(k1 - 1):
            edges.append((i, i + 1))
        if k1 >= 3:
            edges.append((0, k1 - 1))   # cycle inside component 1
        edges.append((k1, k1 + 1))      # component 2 is a pair
        self.edges = edges
        # "shortest_path_distances" chains BFS traversal → distance array (GRD=4):
        # must know which BFS layer a node is discovered on to know its distance.
        self.variant = random.choices(
            ["num_components", "bfs_order", "shortest_path_distances"],
            weights=[2, 2, 3]    # prefer the multi-concept variant
        )[0]
        self.start = 0

    def _build_adj(self):
        adj = {i: [] for i in range(self.n)}
        for u, v in self.edges:
            adj[u].append(v)
            adj[v].append(u)
        for i in adj:
            adj[i].sort()
        return adj

    def solution(self):
        adj = self._build_adj()
        visited = [-1] * self.n
        comp_id = 0
        for start in range(self.n):
            if visited[start] == -1:
                q = deque([start])
                visited[start] = comp_id
                while q:
                    node = q.popleft()
                    for nb in adj[node]:
                        if visited[nb] == -1:
                            visited[nb] = comp_id
                            q.append(nb)
                comp_id += 1

        # BFS order + distances from self.start
        bfs_order = []
        dist = [-1] * self.n
        vis2 = [False] * self.n
        q = deque([self.start])
        vis2[self.start] = True
        dist[self.start] = 0
        while q:
            node = q.popleft()
            bfs_order.append(node)
            for nb in adj[node]:
                if not vis2[nb]:
                    vis2[nb] = True
                    dist[nb] = dist[node] + 1
                    q.append(nb)

        # Reachable nodes and their distances (exclude unreachable -1)
        reachable_dists = {v: dist[v] for v in bfs_order}
        return {
            "num_components":   comp_id,
            "bfs_order":        bfs_order,
            "labels":           visited,
            "distances":        reachable_dists,
            "sum_distances":    sum(reachable_dists.values()),
        }

    def generation(self) -> Dict[str, Any]:
        sol = self.solution()
        edge_str = ", ".join(f"({u},{v})" for u, v in self.edges)
        isolated = [i for i in range(self.n) if not any(i in e for e in self.edges)]
        if self.variant == "num_components":
            question = (
                f"Consider an undirected graph with {self.n} vertices (0 to {self.n-1}) "
                f"and edges: {edge_str}. "
                f"Vertices {isolated} have no edges. "
                f"How many connected components does this graph have?"
            )
            answer = str(sol["num_components"])
        elif self.variant == "bfs_order":
            question = (
                f"Consider an undirected graph with {self.n} vertices (0 to {self.n-1}) "
                f"and edges: {edge_str}. "
                f"Run BFS starting from vertex {self.start}, processing neighbours in ascending order. "
                f"List the vertices reachable from {self.start} in the order they are first visited."
            )
            answer = str(sol["bfs_order"])
        else:   # shortest_path_distances — chains BFS layer → distance (GRD=4)
            question = (
                f"Consider an undirected graph with {self.n} vertices (0 to {self.n-1}) "
                f"and edges: {edge_str}. "
                f"Run BFS from vertex {self.start} (process neighbours in ascending order). "
                f"After BFS completes, what is the SUM of shortest-path distances "
                f"from vertex {self.start} to all vertices reachable from it?"
            )
            answer = str(sol["sum_distances"])
        return {
            "question":  question,
            "answer":    answer,
            "algorithm": "BFS graph traversal",
            "meta":      {"n": self.n, "edges": self.edges, **sol},
        }


# ---------------------------------------------------------------------------
# 8. DFS — Iterative with Stack
# ---------------------------------------------------------------------------

class DFSQuestion(RVQuestion):
    """
    Edge-case: graph with a back-edge (cycle) and an isolated node.
    Question: DFS visit order (iterative, stack pops in reverse alphabetical).
    """

    def initialization(self):
        # Randomise which nodes form the cycle; node (n-1) is always isolated
        # This ensures DFS visit order changes across instances
        templates = [
            [(0,1),(0,2),(1,3),(2,3),(3,4)],   # 3 reachable two ways, cycle 0-1-3-2
            [(0,1),(0,2),(1,2),(2,3),(3,4)],   # triangle at top
            [(0,1),(1,2),(2,3),(0,3),(3,4)],   # square with diagonal
            [(0,2),(0,1),(1,3),(2,3),(2,4)],   # diamond shape
        ]
        self.edges = random.choice(templates)
        self.n     = 6   # node 5 always isolated
        self.start = 0

    def solution(self):
        adj = {i: [] for i in range(self.n)}
        for u, v in self.edges:
            adj[u].append(v); adj[v].append(u)
        for i in adj: adj[i].sort(reverse=True)  # push in reverse order → pop smallest first

        visited = []
        seen = set()
        stack = [self.start]
        while stack:
            node = stack.pop()
            if node in seen: continue
            seen.add(node); visited.append(node)
            for nb in adj[node]:
                if nb not in seen:
                    stack.append(nb)
        return {"dfs_order": visited}

    def generation(self) -> Dict[str, Any]:
        sol = self.solution()
        edge_str = ", ".join(f"({u},{v})" for u, v in self.edges)
        question = (
            f"Perform iterative DFS on an undirected graph with 6 vertices (0-5) "
            f"and edges: {edge_str}. Vertex 5 is isolated. "
            f"Start from vertex {self.start}. When pushing neighbours onto the stack, "
            f"push in descending order (so the smallest-numbered neighbour is popped first). "
            f"List all vertices reachable from {self.start} in the order they are first visited."
        )
        return {
            "question":  question,
            "answer":    str(sol["dfs_order"]),
            "algorithm": "DFS graph traversal",
            "meta":      {"edges": self.edges, **sol},
        }


# ---------------------------------------------------------------------------
# 9. 0/1 Knapsack DP
# ---------------------------------------------------------------------------

class KnapsackDPQuestion(RVQuestion):
    """
    Edge-case: weights designed so that the optimal solution is NOT simply
    taking the highest-value item (forces DP table reasoning).
    """

    def initialization(self):
        n = random.randint(3, 5)
        W = random.randint(5, 10)
        self.W = W
        # Design adversarial items: one high-value heavy item, several lighter ones
        heavy_w = W - 1
        heavy_v = random.randint(5, 8)
        items = [(heavy_w, heavy_v)]
        for _ in range(n - 1):
            w = random.randint(1, W // 2)
            v = random.randint(1, heavy_v)
            items.append((w, v))
        random.shuffle(items)
        self.weights = [it[0] for it in items]
        self.values  = [it[1] for it in items]
        self.variant = random.choice(["max_value", "dp_cell"])
        if self.variant == "dp_cell":
            # Ask about a specific cell in the DP table
            self.i_ask = random.randint(1, n)
            self.w_ask = random.randint(1, W)

    def solution(self):
        n = len(self.weights)
        dp = [[0] * (self.W + 1) for _ in range(n + 1)]
        for i in range(1, n + 1):
            for w in range(self.W + 1):
                dp[i][w] = dp[i-1][w]
                if self.weights[i-1] <= w:
                    dp[i][w] = max(dp[i][w], dp[i-1][w - self.weights[i-1]] + self.values[i-1])
        return {"max_value": dp[n][self.W], "dp_table": dp}

    def generation(self) -> Dict[str, Any]:
        sol = self.solution()
        items_str = ", ".join(f"(w={w}, v={v})" for w, v in zip(self.weights, self.values))
        if self.variant == "max_value":
            question = (
                f"Solve the 0/1 knapsack problem. Items: {items_str}. "
                f"Knapsack capacity W = {self.W}. "
                f"What is the maximum total value achievable?"
            )
            answer = str(sol["max_value"])
        else:
            cell = sol["dp_table"][self.i_ask][self.w_ask]
            question = (
                f"Fill the 0/1 knapsack DP table. Items: {items_str}. "
                f"Capacity W = {self.W}. "
                f"What is the value of dp[{self.i_ask}][{self.w_ask}]?"
            )
            answer = str(cell)
        return {
            "question":  question,
            "answer":    answer,
            "algorithm": "0/1 knapsack dynamic programming",
            "meta": {
                "weights": self.weights, "values": self.values,
                "W": self.W, "max_value": sol["max_value"],
            },
        }


# ---------------------------------------------------------------------------
# 10. LCS DP
# ---------------------------------------------------------------------------

class LCSDPQuestion(RVQuestion):
    """
    Edge-case: strings that are reverses of each other (LCS ≠ length of strings).
    """

    def initialization(self):
        length = random.randint(3, 5)
        chars  = random.sample("ABCDE", length)
        self.X = "".join(chars)
        # Reverse + minor perturbation: maximises non-obvious LCS
        self.Y = "".join(reversed(chars))
        if random.random() < 0.5:
            # Swap two adjacent characters in Y for extra complexity
            i = random.randint(0, length - 2)
            yl = list(self.Y)
            yl[i], yl[i+1] = yl[i+1], yl[i]
            self.Y = "".join(yl)

    def solution(self):
        m, n = len(self.X), len(self.Y)
        dp = [[0]*(n+1) for _ in range(m+1)]
        for i in range(1, m+1):
            for j in range(1, n+1):
                if self.X[i-1] == self.Y[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return {"lcs_length": dp[m][n], "dp": dp}

    def generation(self) -> Dict[str, Any]:
        sol = self.solution()
        question = (
            f'Apply the LCS dynamic programming algorithm to strings '
            f'X = "{self.X}" and Y = "{self.Y}". '
            f'What is the length of the longest common subsequence?'
        )
        return {
            "question":  question,
            "answer":    str(sol["lcs_length"]),
            "algorithm": "LCS dynamic programming",
            "meta":      {"X": self.X, "Y": self.Y, "lcs_length": sol["lcs_length"]},
        }


# ---------------------------------------------------------------------------
# 11. Recursion — Tower of Hanoi + Triangular Numbers
# ---------------------------------------------------------------------------

class RecursionQuestion(RVQuestion):
    """
    Variants:
      hanoi_moves  — total moves for n discs
      hanoi_state  — state of pegs after k moves
      triangular   — T(n) using recursive formula
      call_depth   — maximum call stack depth for a given recursive function
    """

    def initialization(self):
        # Weights: prefer call_depth and triangular (GRD=3-4) over hanoi_moves
        # hanoi_moves with small n is T(n)=2^n-1 — one formula, GRD=2.
        # call_depth requires reasoning about the full stack, GRD=3.
        # triangular requires tracing n additions and unwinding, GRD=3.
        self.variant = random.choices(
            ["hanoi_moves", "triangular", "call_depth"],
            weights=[1, 3, 3]     # down-weight hanoi_moves
        )[0]
        if self.variant == "hanoi_moves":
            self.n = random.randint(4, 6)   # larger n → slightly more complex
        elif self.variant == "triangular":
            self.n = random.randint(5, 9)
        else:  # call_depth
            self.n = random.randint(5, 8)

    def solution(self):
        if self.variant == "hanoi_moves":
            return {"answer": 2**self.n - 1}
        elif self.variant == "triangular":
            return {"answer": self.n * (self.n + 1) // 2}
        else:
            return {"answer": self.n}  # call stack depth = n for T(n)

    def generation(self) -> Dict[str, Any]:
        sol = self.solution()
        if self.variant == "hanoi_moves":
            question = (
                f"The Tower of Hanoi algorithm moves n discs from source to target peg "
                f"using T(n) = 2·T(n-1) + 1 total moves, with T(0) = 0. "
                f"How many total moves are required to move {self.n} discs?"
            )
        elif self.variant == "triangular":
            question = (
                f"Consider the recursive function: T(1) = 1, T(n) = n + T(n-1) for n > 1. "
                f"Trace the complete call sequence and compute T({self.n})."
            )
        else:
            question = (
                f"For the recursive function T(n) = T(n-1) + n with base case T(1) = 1, "
                f"how many stack frames are on the call stack at the deepest point "
                f"during the computation of T({self.n})?"
            )
        return {
            "question":  question,
            "answer":    str(sol["answer"]),
            "algorithm": "recursion",
            "meta":      {"n": self.n, "variant": self.variant, **sol},
        }


# ---------------------------------------------------------------------------
# Topic → class mapping (used by SmartGenerator)
# ---------------------------------------------------------------------------

TOPIC_TO_RV_CLASSES = {
    "hash table":          [HashTableLinearProbingQuestion, HashTableChainingQuestion],
    "hash table linear probing":   [HashTableLinearProbingQuestion],
    "hash table quadratic probing":[HashTableLinearProbingQuestion],  # LP is close enough
    "sorting algorithm":   [InsertionSortQuestion, BubbleSortStabilityQuestion, MergeSortQuestion],
    "sorting":             [InsertionSortQuestion, BubbleSortStabilityQuestion, MergeSortQuestion],
    "graph traversal":     [BFSComponentsQuestion, DFSQuestion],
    "graph":               [BFSComponentsQuestion, DFSQuestion],
    "dynamic programming": [KnapsackDPQuestion, LCSDPQuestion],
    "recursion":           [RecursionQuestion],
    "binary search tree":  [BSTOperationsQuestion],
    "bst":                 [BSTOperationsQuestion],
}


def get_rv_class(topic: str):
    """Return a random RV question class for the given topic, or None."""
    t = topic.lower().strip()
    for key, classes in TOPIC_TO_RV_CLASSES.items():
        if key in t:
            return random.choice(classes)
    return None


def generate_rv_question(topic: str) -> Dict[str, Any]:
    """
    Convenience function: pick an RV class for `topic`, instantiate,
    sample random variables, and return the full QA dict.
    Returns None if topic has no RV class.
    """
    cls = get_rv_class(topic)
    if cls is None:
        return None
    return cls().sample()
