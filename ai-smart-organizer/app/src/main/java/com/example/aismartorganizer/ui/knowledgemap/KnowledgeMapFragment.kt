package com.example.aismartorganizer.ui.knowledgemap

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.Fragment
import androidx.fragment.app.activityViewModels
import androidx.recyclerview.widget.LinearLayoutManager
import com.example.aismartorganizer.SmartOrganizerApp
import com.example.aismartorganizer.adapter.TreeAdapter
import com.example.aismartorganizer.databinding.FragmentKnowledgeMapBinding
import com.example.aismartorganizer.utils.DataStructureUtils
import com.example.aismartorganizer.viewmodel.MainViewModel
import com.example.aismartorganizer.viewmodel.ViewModelFactory

class KnowledgeMapFragment : Fragment() {
    private var _binding: FragmentKnowledgeMapBinding? = null
    private val binding get() = _binding!!

    private val viewModel: MainViewModel by activityViewModels {
        ViewModelFactory((requireActivity().application as SmartOrganizerApp).repository)
    }

    private val adapter = TreeAdapter()

    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?): View {
        _binding = FragmentKnowledgeMapBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        binding.treeRecycler.layoutManager = LinearLayoutManager(requireContext())
        binding.treeRecycler.adapter = adapter

        viewModel.notes.observe(viewLifecycleOwner) {
            val flattened = DataStructureUtils.flattenTree(it)
            adapter.submitTree(flattened)
            binding.graphHint.text = "Graph edges: ${DataStructureUtils.buildRelatedGraph(it).size} nodes linked"
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}
