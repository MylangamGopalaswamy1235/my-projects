package com.example.aismartorganizer.ui.insights

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.Fragment
import androidx.fragment.app.activityViewModels
import com.example.aismartorganizer.SmartOrganizerApp
import com.example.aismartorganizer.databinding.FragmentInsightsBinding
import com.example.aismartorganizer.utils.InsightsAnalyzer
import com.example.aismartorganizer.viewmodel.MainViewModel
import com.example.aismartorganizer.viewmodel.ViewModelFactory

class InsightsFragment : Fragment() {

    private var _binding: FragmentInsightsBinding? = null
    private val binding get() = _binding!!

    private val viewModel: MainViewModel by activityViewModels {
        ViewModelFactory((requireActivity().application as SmartOrganizerApp).repository)
    }

    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?): View {
        _binding = FragmentInsightsBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        val observer = {
            val result = InsightsAnalyzer.analyze(
                viewModel.notes.value.orEmpty(),
                viewModel.tasks.value.orEmpty()
            )
            binding.passwordStats.text = "Total: ${result.totalPasswords} | Strong: ${result.strongPasswords} | Weak: ${result.weakPasswords} | Duplicate: ${result.duplicatePasswords}"
            binding.wordStats.text = "Most used word: ${result.mostUsedWord}"
            val total = (result.completedTasks + result.pendingTasks).coerceAtLeast(1)
            binding.progressTasks.progress = (result.completedTasks * 100) / total
            binding.taskStats.text = "Completed: ${result.completedTasks}, Pending: ${result.pendingTasks}"
        }

        viewModel.notes.observe(viewLifecycleOwner) { observer() }
        viewModel.tasks.observe(viewLifecycleOwner) { observer() }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}
